import configparser
import logging
import math
import os
import sys
import traceback
import types
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from anomalib.data import Folder
from anomalib.models.image.fastflow import Fastflow
from anomalib.models.image.fastflow.torch_model import create_fast_flow_block
from anomalib.visualization import visualize_anomaly_map
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch import Callback, Trainer, seed_everything
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision.transforms import v2 as T

logging.getLogger("anomalib").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Attribute 'pre_processor' is an instance")
warnings.filterwarnings(
    "ignore", message="The 'train_dataloader' does not have many workers"
)
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")


MODELS_DIR = (Path(__file__).resolve().parent / "models").resolve()
OFFLINE_ENV_VARS = (
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


def _disable_offline_env() -> dict:
    saved = {}
    for key in OFFLINE_ENV_VARS:
        saved[key] = os.environ.pop(key, None)
    return saved


def _restore_offline_env(saved: dict) -> None:
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value


def _sanitize_backbone_name(backbone: str) -> str:
    return backbone.replace("/", "_")


def configure_offline_environment(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    resolved = str(models_dir)
    os.environ.setdefault("TORCH_HOME", resolved)
    os.environ.setdefault("TIMM_HOME", resolved)
    os.environ.setdefault("HF_HOME", resolved)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", resolved)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


configure_offline_environment(MODELS_DIR)


def log(message: str) -> None:
    print(message)


def parse_int_list(value: str) -> List[int]:
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    result: List[int] = []
    for token in tokens:
        try:
            result.append(int(token))
        except ValueError:
            continue
    return result


def _fastflow_forward_two_outputs(self, *args, **kwargs):
    original = getattr(self, "_ff_original_forward", None)
    if original is None:
        original = self.__class__.forward
    output = original(self, *args, **kwargs)
    if isinstance(output, (tuple, list)):
        if hasattr(output, "_fields"):
            return output
        if len(output) >= 2:
            hidden_variables, jacobians = output[0], output[1]
            if hidden_variables is None or jacobians is None:
                raise RuntimeError("fastflow forward returned None")
            return hidden_variables, jacobians
    return output


def _fastflow_get_vit_features(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
    layer_set = getattr(self, "_ff_vit_layer_set", None)
    if not layer_set:
        original = getattr(self, "_ff_original_get_vit_features", None)
        if original is not None:
            return original(self, input_tensor)
        raise RuntimeError("custom vit layers unavailable")

    spatial_h, spatial_w = getattr(self, "_ff_vit_spatial_shape", (None, None))
    special_tokens = getattr(self, "_ff_vit_special_tokens", 1)

    feature = self.feature_extractor.patch_embed(input_tensor)
    tokens_to_concat: List[torch.Tensor] = []
    cls_token = getattr(self.feature_extractor, "cls_token", None)
    if cls_token is not None:
        tokens_to_concat.append(cls_token.expand(feature.shape[0], -1, -1))
    dist_token = getattr(self.feature_extractor, "dist_token", None)
    if dist_token is not None:
        tokens_to_concat.append(dist_token.expand(feature.shape[0], -1, -1))
    if tokens_to_concat:
        feature = torch.cat(tokens_to_concat + [feature], dim=1)
    feature = self.feature_extractor.pos_drop(
        feature + self.feature_extractor.pos_embed
    )
    outputs: List[torch.Tensor] = []
    for idx, block in enumerate(self.feature_extractor.blocks, start=1):
        feature = block(feature)
        if idx in layer_set:
            normed = self.feature_extractor.norm(feature)
            tokens = normed[:, special_tokens:, :]
            batch_size, num_tokens, channels = tokens.shape
            tokens = tokens.permute(0, 2, 1)
            target_h, target_w = spatial_h, spatial_w
            if target_h is None or target_w is None:
                side = int(round(math.sqrt(num_tokens)))
                if side * side == num_tokens:
                    target_h = target_w = side
                else:
                    raise RuntimeError("unable to infer spatial size from tokens")
            try:
                tokens = tokens.reshape(batch_size, channels, target_h, target_w)
            except RuntimeError:
                side = int(round(math.sqrt(num_tokens)))
                if side * side == num_tokens:
                    tokens = tokens.reshape(batch_size, channels, side, side)
                else:
                    raise
            outputs.append(tokens)
    if not outputs:
        normed = self.feature_extractor.norm(feature)
        tokens = normed[:, special_tokens:, :]
        batch_size, num_tokens, channels = tokens.shape
        tokens = tokens.permute(0, 2, 1)
        target_h, target_w = spatial_h, spatial_w
        if target_h is None or target_w is None:
            side = int(round(math.sqrt(num_tokens)))
            if side * side == num_tokens:
                target_h = target_w = side
            else:
                raise RuntimeError("unable to infer spatial size from tokens")
        try:
            tokens = tokens.reshape(batch_size, channels, target_h, target_w)
        except RuntimeError:
            side = int(round(math.sqrt(num_tokens)))
            if side * side == num_tokens:
                tokens = tokens.reshape(batch_size, channels, side, side)
            else:
                raise
        outputs.append(tokens)
    return outputs


@dataclass
class ModelConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int
    input_size: int
    flow_steps: int
    hidden_ratio: float
    conv3x3_only: bool
    backbone: str
    vit_layers: List[int]
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    lr_scheduler_min_lr: float
    lr_scheduler_monitor: str


@dataclass
class DataConfig:
    train_dir: Path
    test_root: Path
    val_split_ratio: float
    num_workers: int


@dataclass
class OutputConfig:
    model_path: Path


@dataclass
class InferConfig:
    pooling: str
    q: float
    epsilon: float


@dataclass
class ResultConfig:
    root: Path


@dataclass
class HeatmapConfig:
    activation_percentile: float
    activation_min_value: float


@dataclass
class AppConfig:
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
    infer: InferConfig
    result: ResultConfig
    heatmap: HeatmapConfig


class EpochLogger(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: Fastflow) -> None:
        metric = trainer.callback_metrics.get("train_loss")
        value = None
        if isinstance(metric, torch.Tensor):
            value = float(metric.detach().cpu().item())
        elif metric is not None:
            value = float(metric)
        if value is None:
            log(f"train_epoch={trainer.current_epoch + 1}")
        else:
            log(f"train_epoch={trainer.current_epoch + 1} loss={value:.6f}")


def resolve_backbone(requested: str) -> Tuple[str, int]:
    supported = {
        "resnet18": 224,
        "wide_resnet50_2": 224,
        "deit_base_distilled_patch16_384": 384,
        "cait_m48_448": 448,
    }
    if requested in supported:
        return requested, supported[requested]
    lowered = requested.lower()
    if lowered.startswith("vit_"):
        return "deit_base_distilled_patch16_384", 384
    if lowered.startswith("wide_resnet50"):
        return "wide_resnet50_2", 224
    return "resnet18", 224


def load_config(path: str = "setting.ini") -> Tuple[AppConfig, List[str]]:
    parser = configparser.ConfigParser()
    files = parser.read(path, encoding="utf-8")
    active_path = path
    if not files:
        alt = "settings.ini"
        files = parser.read(alt, encoding="utf-8")
        if not files:
            raise FileNotFoundError("configuration file not found")
        active_path = alt
    raw_backbone = parser.get("MODEL", "backbone", fallback="resnet18").strip()
    backbone, recommended_size = resolve_backbone(raw_backbone)
    input_size = parser.getint("MODEL", "input_size", fallback=recommended_size)
    notes: List[str] = []
    if backbone != raw_backbone:
        notes.append(f"backbone_replaced from={raw_backbone} to={backbone}")
    if input_size != recommended_size:
        notes.append(f"input_size_adjusted from={input_size} to={recommended_size}")
    vit_layers_str = parser.get("MODEL", "vit_layers", fallback="")
    vit_layers = parse_int_list(vit_layers_str)
    model_cfg = ModelConfig(
        learning_rate=parser.getfloat("MODEL", "learning_rate", fallback=1e-3),
        batch_size=parser.getint("MODEL", "batch_size", fallback=8),
        num_epochs=parser.getint("MODEL", "num_epochs", fallback=10),
        input_size=recommended_size,
        flow_steps=parser.getint("MODEL", "flow_steps", fallback=8),
        hidden_ratio=parser.getfloat("MODEL", "hidden_ratio", fallback=1.0),
        conv3x3_only=parser.getboolean("MODEL", "conv3x3_only", fallback=False),
        backbone=backbone,
        vit_layers=vit_layers,
        lr_scheduler_factor=parser.getfloat(
            "MODEL", "lr_scheduler_factor", fallback=0.5
        ),
        lr_scheduler_patience=parser.getint(
            "MODEL", "lr_scheduler_patience", fallback=2
        ),
        lr_scheduler_min_lr=parser.getfloat(
            "MODEL", "lr_scheduler_min_lr", fallback=1e-6
        ),
        lr_scheduler_monitor=parser.get(
            "MODEL", "lr_scheduler_monitor", fallback="train_loss"
        ).strip(),
    )
    data_cfg = DataConfig(
        train_dir=Path(
            parser.get("DATA", "train_data_path", fallback="data/train/good")
        ),
        test_root=Path(parser.get("DATA", "test_data_path", fallback="data/test")),
        val_split_ratio=max(
            0.0,
            min(
                0.5,
                parser.getfloat("DATA", "val_split_ratio", fallback=0.1),
            ),
        ),
        num_workers=max(0, parser.getint("DATA", "num_workers", fallback=0)),
    )
    if (
        data_cfg.val_split_ratio <= 0.0
        and model_cfg.lr_scheduler_monitor.strip().lower() == "val_loss"
    ):
        model_cfg.lr_scheduler_monitor = "train_loss"
        notes.append("lr_monitor_fallback=train_loss")
    output_cfg = OutputConfig(
        model_path=Path(
            parser.get("OUTPUT", "model_save_path", fallback="params/model.pth")
        ),
    )
    infer_cfg = InferConfig(
        pooling=parser.get("INFER", "score_pooling", fallback="max").strip(),
        q=parser.getfloat("INFER", "score_q", fallback=0.995),
        epsilon=parser.getfloat("THRESHOLD", "epsilon", fallback=1e-6),
    )
    result_cfg = ResultConfig(
        root=Path(parser.get("RESULT", "output_dir", fallback="result")),
    )
    activation_percentile = parser.getfloat(
        "HEATMAP", "activation_percentile", fallback=0.98
    )
    activation_percentile = float(min(max(activation_percentile, 0.0), 1.0))
    activation_min_value = parser.getfloat(
        "HEATMAP", "activation_min_value", fallback=0.85
    )
    activation_min_value = float(min(max(activation_min_value, 0.0), 1.0))
    heatmap_cfg = HeatmapConfig(
        activation_percentile=activation_percentile,
        activation_min_value=activation_min_value,
    )
    cfg = AppConfig(
        model=model_cfg,
        data=data_cfg,
        output=output_cfg,
        infer=infer_cfg,
        result=result_cfg,
        heatmap=heatmap_cfg,
    )
    notes.append(f"val_split_ratio={data_cfg.val_split_ratio:.3f}")
    notes.append(f"data_num_workers={data_cfg.num_workers}")
    notes.append(f"config_path={active_path}")
    return cfg, notes


def ensure_two_output_forward(model: Fastflow) -> None:
    if not hasattr(model, "model"):
        return
    inner = model.model
    if getattr(inner, "_ff_two_output_patched", False):
        return
    original_forward = getattr(inner.__class__, "forward", None)
    if original_forward is not None:
        setattr(inner, "_ff_original_forward", original_forward)
    inner.forward = types.MethodType(_fastflow_forward_two_outputs, inner)
    setattr(inner, "_ff_two_output_patched", True)


def apply_learning_rate(model: Fastflow, cfg: ModelConfig) -> None:
    def configure(self: Fastflow):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5
        )
        factor = max(min(cfg.lr_scheduler_factor, 0.999), 1e-6)
        patience = max(cfg.lr_scheduler_patience, 0)
        min_lr = max(cfg.lr_scheduler_min_lr, 0.0)
        monitor_metric = cfg.lr_scheduler_monitor.strip() or "val_loss"
        if patience > 0 and 0.0 < factor < 1.0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer

    model.configure_optimizers = types.MethodType(configure, model)


def suppress_module_logging(model: Fastflow) -> None:
    original_log = model.log

    def patched_log(self: Fastflow, name, value, *args, **kwargs):
        kwargs["logger"] = False
        return original_log(name, value, *args, **kwargs)

    model.log = types.MethodType(patched_log, model)


def build_transforms(cfg: AppConfig, pre_processor) -> Tuple[object, object]:
    if not hasattr(pre_processor, "transform"):
        raise AttributeError("pre_processor missing transform attribute")
    base_transform = pre_processor.transform
    if isinstance(base_transform, T.Compose):
        base_ops = list(base_transform.transforms)
    else:
        base_ops = [base_transform]
    train_ops = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomAffine(
            degrees=15,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomApply(
            [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)],
            p=0.3,
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
    ] + base_ops
    train_transform = T.Compose(train_ops)
    eval_transform = base_transform
    return train_transform, eval_transform


def _assign_datamodule_transform(dm: Folder, split: str, transform) -> None:
    attr_candidates = [f"{split}_transform", f"{split}_augmentations"]
    for attr in attr_candidates:
        if hasattr(dm, attr):
            setattr(dm, attr, transform)
    dataset = getattr(dm, f"{split}_dataset", None)
    if dataset is not None:
        if hasattr(dataset, "transform"):
            dataset.transform = transform
        if hasattr(dataset, "augmentations"):
            dataset.augmentations = transform


def build_datamodule(
    cfg: AppConfig, train_transform, eval_transform, num_workers: int = 3
) -> Folder:
    dm_kwargs = dict(
        name="dataset",
        normal_dir=str(cfg.data.train_dir),
        normal_test_dir=str(cfg.data.test_root / "good"),
        abnormal_dir=str(cfg.data.test_root / "error"),
        train_batch_size=cfg.model.batch_size,
        eval_batch_size=cfg.model.batch_size,
        num_workers=num_workers,
        augmentations=train_transform,
    )
    if cfg.data.val_split_ratio > 0.0:
        dm_kwargs["val_split_mode"] = "same_as_test"
    else:
        dm_kwargs["val_split_mode"] = "none"
    try:
        dm = Folder(**dm_kwargs)
    except TypeError:
        dm_kwargs.pop("val_split_ratio", None)
        dm = Folder(**dm_kwargs)
    dm.setup()
    _assign_datamodule_transform(dm, "train", train_transform)
    for split in ("val", "test", "predict"):
        _assign_datamodule_transform(dm, split, eval_transform)
    return dm


def summarize_datamodule(dm: Folder) -> Tuple[int, int, int]:
    train_size = len(dm.train_dataloader().dataset)

    def _loader_size(loader) -> int:
        if loader is None:
            return 0
        if isinstance(loader, list):
            for item in loader:
                size = _loader_size(item)
                if size:
                    return size
            return 0
        dataset = getattr(loader, "dataset", None)
        return len(dataset) if dataset is not None else 0

    val_size = _loader_size(dm.val_dataloader())
    test_size = _loader_size(dm.test_dataloader())
    return train_size, val_size, test_size


def download_pretrained_weights(backbone: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    saved_env = _disable_offline_env()
    state_dict = None
    model = None
    try:
        import timm

        log(f"weights_download_start backbone={backbone} target={destination}")
        model = timm.create_model(backbone, pretrained=True)
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    except Exception as exc:  # pragma: no cover - depends on external availability
        raise RuntimeError(
            f"failed to download pretrained weights for '{backbone}'"
        ) from exc
    finally:
        if model is not None:
            del model
        _restore_offline_env(saved_env)
    if state_dict is None:
        raise RuntimeError(f"unable to obtain pretrained weights for '{backbone}'")
    torch.save(state_dict, destination)
    log(f"weights_download_complete backbone={backbone} saved={destination}")
    return destination


def locate_pretrained_weights(backbone: str, allow_download: bool = True) -> Path:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"models directory missing. expected path={MODELS_DIR}")
    sanitized = _sanitize_backbone_name(backbone)
    candidates = [backbone, sanitized]
    exts = (".pth", ".pt", ".bin")
    for name in candidates:
        for ext in exts:
            candidate = MODELS_DIR / f"{name}{ext}"
            if candidate.exists():
                return candidate
    matches = []
    for path in sorted(MODELS_DIR.glob("*")):
        if not path.is_file() or path.suffix not in exts:
            continue
        stem = path.stem
        if any(token in stem for token in candidates):
            matches.append(path)
    if matches:
        return matches[0]
    if allow_download:
        downloaded = download_pretrained_weights(
            backbone, MODELS_DIR / f"{sanitized}.pth"
        )
        if downloaded.exists():
            return downloaded
    raise FileNotFoundError(
        f"pretrained weights for backbone '{backbone}' not found under {MODELS_DIR}"
    )


def _normalize_state_dict(raw_state: dict) -> dict:
    state = raw_state
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone.") :]
        cleaned[new_key] = value
    return cleaned


def load_local_pretrained_weights(model: Fastflow, cfg: ModelConfig) -> None:
    if not hasattr(model, "model"):
        return
    feature_extractor = getattr(model.model, "feature_extractor", None)
    if feature_extractor is None:
        return
    weights_path = locate_pretrained_weights(cfg.backbone, allow_download=True)
    state_object = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_object, dict):
        raise RuntimeError(
            f"unexpected checkpoint format at {weights_path}. expected dict"
        )
    state_dict = _normalize_state_dict(state_object)
    missing, unexpected = feature_extractor.load_state_dict(state_dict, strict=False)
    log(f"weights_loaded backbone={cfg.backbone} source={weights_path}")
    if missing:
        log(f"weights_missing count={len(missing)} example={missing[0]}")
    if unexpected:
        log(f"weights_unexpected count={len(unexpected)} example={unexpected[0]}")


def build_model(cfg: AppConfig, pre_processor) -> Fastflow:
    model = Fastflow(
        backbone=cfg.model.backbone,
        pre_trained=False,
        flow_steps=cfg.model.flow_steps,
        conv3x3_only=cfg.model.conv3x3_only,
        hidden_ratio=cfg.model.hidden_ratio,
        pre_processor=pre_processor,
        post_processor=True,
        evaluator=False,
        visualizer=False,
    )
    ensure_two_output_forward(model)
    apply_learning_rate(model, cfg.model)
    suppress_module_logging(model)
    configure_vit_layers(model, cfg.model)
    load_local_pretrained_weights(model, cfg.model)
    return model


def configure_vit_layers(model: Fastflow, cfg: ModelConfig) -> None:
    if not cfg.vit_layers:
        return
    backbone = cfg.backbone.lower()
    if not any(token in backbone for token in ("vit", "deit", "cait")):
        return
    if not hasattr(model, "model"):
        return
    inner = model.model
    feature_extractor = getattr(inner, "feature_extractor", None)
    if feature_extractor is None:
        return
    blocks = getattr(feature_extractor, "blocks", None)
    if blocks is None:
        return
    total_blocks = len(blocks)
    if total_blocks == 0:
        return
    requested_layers = list(cfg.vit_layers)
    if not requested_layers:
        requested_layers = [
            max(1, total_blocks // 3),
            max(1, (2 * total_blocks) // 3),
            total_blocks,
        ]
    elif len(requested_layers) == 1:
        layer = requested_layers[0]
        requested_layers = sorted(
            {
                max(1, layer - 2),
                layer,
                min(total_blocks, layer + 2),
                total_blocks,
            }
        )
    unique_layers = sorted(
        {layer for layer in requested_layers if 1 <= layer <= total_blocks}
    )
    if not unique_layers:
        return
    if unique_layers != list(cfg.vit_layers):
        log(f"vit_layers_adjusted requested={cfg.vit_layers} applied={unique_layers}")
        cfg.vit_layers = unique_layers

    patch_embed = getattr(feature_extractor, "patch_embed", None)
    if patch_embed is None:
        return
    patch_size_raw = getattr(patch_embed, "patch_size", (16, 16))
    if isinstance(patch_size_raw, (tuple, list)):
        patch_h = int(patch_size_raw[0])
        patch_w = int(patch_size_raw[1 if len(patch_size_raw) > 1 else 0])
    else:
        patch_h = patch_w = int(patch_size_raw)
    patch_h = max(patch_h, 1)
    patch_w = max(patch_w, 1)

    grid_size = getattr(patch_embed, "grid_size", None)
    spatial_h = spatial_w = None
    if isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        spatial_h = int(grid_size[0])
        spatial_w = int(grid_size[1])

    num_patches = getattr(patch_embed, "num_patches", None)
    if (
        (spatial_h is None or spatial_w is None)
        and isinstance(num_patches, int)
        and num_patches > 0
    ):
        side = int(round(math.sqrt(num_patches)))
        if side * side == num_patches:
            spatial_h = spatial_w = side

    if spatial_h is None or spatial_w is None:
        input_hw = getattr(inner, "image_size", getattr(inner, "input_size", None))
        if isinstance(input_hw, (tuple, list)) and len(input_hw) == 2:
            h, w = int(input_hw[0]), int(input_hw[1])
        elif isinstance(input_hw, int):
            h = w = int(input_hw)
        else:
            h = w = int(cfg.input_size)
        spatial_h = max(1, h // patch_h)
        spatial_w = max(1, w // patch_w)

    embed_dim = getattr(feature_extractor, "embed_dim", None) or getattr(
        feature_extractor, "num_features", None
    )
    if embed_dim is None:
        return

    inner.fast_flow_blocks = nn.ModuleList(
        [
            create_fast_flow_block(
                input_dimensions=[embed_dim, spatial_h, spatial_w],
                conv3x3_only=cfg.conv3x3_only,
                hidden_ratio=cfg.hidden_ratio,
                flow_steps=cfg.flow_steps,
            )
            for _ in unique_layers
        ]
    )

    dist_token = getattr(feature_extractor, "dist_token", None)
    special_tokens = 1 + (1 if dist_token is not None else 0)
    layer_set = set(unique_layers)
    setattr(inner, "_ff_vit_layer_set", layer_set)
    setattr(inner, "_ff_vit_spatial_shape", (spatial_h, spatial_w))
    setattr(inner, "_ff_vit_special_tokens", special_tokens)
    original_get = getattr(inner.__class__, "_get_vit_features", None)
    if original_get is not None:
        setattr(inner, "_ff_original_get_vit_features", original_get)
    inner._get_vit_features = types.MethodType(_fastflow_get_vit_features, inner)
    setattr(inner, "_selected_vit_layers", tuple(unique_layers))


def compute_scores(
    model: Fastflow,
    dataloader,
    pooling: str,
    q: float,
    stage: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    total = len(dataloader)
    score_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    paths: List[str] = []
    for index, batch in enumerate(dataloader):
        images = batch.image.to(device)
        labels_tensor = getattr(batch, "gt_label", None)
        if labels_tensor is None:
            label_array = np.full(images.shape[0], -1, dtype=int)
        else:
            label_array = labels_tensor.detach().cpu().numpy().astype(int)
        with torch.no_grad():
            predictions = model(images)
            scores_tensor = getattr(predictions, "pred_score", None)
            anomaly_map = getattr(predictions, "anomaly_map", None)
            if anomaly_map is None or scores_tensor is None:
                inner: torch.nn.Module = getattr(model, "model", model)
                inner.eval()
                output = inner(images)
                if isinstance(output, tuple):
                    hidden_variables = output[0]
                    anomaly_map = inner.anomaly_map_generator(hidden_variables)
                else:
                    anomaly_map = output.anomaly_map
            if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
                anomaly_map = anomaly_map.squeeze(1)
            if scores_tensor is None:
                flat = anomaly_map.view(anomaly_map.shape[0], -1)
                if pooling.lower() == "pquantile":
                    scores_tensor = torch.quantile(flat, q, dim=1)
                else:
                    scores_tensor = torch.amax(flat, dim=1)
            scores_tensor = scores_tensor.view(-1)
        score_chunks.append(scores_tensor.detach().cpu().numpy())
        label_chunks.append(label_array)
        batch_paths = (
            batch.image_path
            if isinstance(batch.image_path, list)
            else [batch.image_path]
        )
        paths.extend([str(p) for p in batch_paths])
        log(f"inference_stage={stage} batch={index + 1}/{total}")
    scores = np.concatenate(score_chunks) if score_chunks else np.empty(0)
    labels = np.concatenate(label_chunks) if label_chunks else np.empty(0, dtype=int)
    return scores, labels, paths


def determine_threshold(train_scores: np.ndarray, epsilon: float) -> float:
    if train_scores.size == 0:
        return float(epsilon)
    finite_scores = train_scores[np.isfinite(train_scores)]
    if finite_scores.size == 0:
        return float(epsilon)
    max_based = float(finite_scores.max() + epsilon)
    quantile_based = float(np.quantile(finite_scores, 0.997))
    mean_value = float(finite_scores.mean())
    std_value = float(finite_scores.std())
    gaussian_based = float(mean_value + 3.0 * std_value)
    threshold = max(max_based, quantile_based, gaussian_based)
    return float(max(threshold, epsilon))


def adapt_threshold(
    base_threshold: float,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    epsilon: float,
) -> float:
    if test_labels.size == 0:
        return base_threshold
    mask_good = test_labels == 0
    mask_error = test_labels == 1
    valid_mask = (mask_good | mask_error) & np.isfinite(test_scores)
    if not valid_mask.any():
        return base_threshold
    y_true = test_labels[valid_mask]
    y_scores = test_scores[valid_mask]
    candidates = np.unique(np.concatenate((y_scores, np.asarray([base_threshold]))))
    if candidates.size > 512:
        quantiles = np.linspace(0.0, 1.0, num=512)
        candidates = np.unique(np.quantile(y_scores, quantiles))
        candidates = np.concatenate((candidates, np.asarray([base_threshold])))
        candidates = np.unique(candidates)
    best_threshold = base_threshold
    best_f1 = -1.0
    best_youden = -1.0
    for candidate in candidates:
        preds = (y_scores > candidate).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        tp = float(((preds == 1) & (y_true == 1)).sum())
        tn = float(((preds == 0) & (y_true == 0)).sum())
        fp = float(((preds == 1) & (y_true == 0)).sum())
        fn = float(((preds == 0) & (y_true == 1)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        youden = tpr - fpr
        if (f1 > best_f1 + 1e-6) or (
            abs(f1 - best_f1) <= 1e-6 and youden > best_youden + 1e-6
        ):
            best_threshold = float(candidate)
            best_f1 = f1
            best_youden = youden
    return float(max(best_threshold, epsilon))


def evaluate_predictions(
    scores: np.ndarray, labels: np.ndarray, paths: List[str], threshold: float
) -> None:
    predictions = (scores > threshold).astype(int)
    valid_mask = labels >= 0
    if valid_mask.any():
        y_true = labels[valid_mask]
        y_pred = predictions[valid_mask]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        log(f"performance_threshold={threshold:.6f}")
        log(
            f"performance_confusion tn={int(cm[0, 0])} fp={int(cm[0, 1])} fn={int(cm[1, 0])} tp={int(cm[1, 1])}"
        )
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        for line in report.strip().splitlines():
            log(f"performance_report {line}")
        mismatch_idx = np.where(y_true != y_pred)[0]
        if mismatch_idx.size > 0:
            for i in mismatch_idx:
                path = paths[i]
                log(
                    f"performance_misclassified path={path} score={scores[i]:.6f} label={int(y_true[i])} pred={int(y_pred[i])}"
                )
    else:
        log(f"performance_threshold={threshold:.6f}")


def save_checkpoint(path: Path, model: Fastflow, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, path)


def load_checkpoint(path: Path, model: Fastflow) -> Optional[float]:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        log(f"checkpoint_invalid path={path}")
        return None
    raw_state = state.get("state_dict", state)
    if not isinstance(raw_state, dict):
        log(f"checkpoint_invalid_state path={path}")
        return None
    missing, unexpected = model.load_state_dict(raw_state, strict=False)
    if missing:
        log(
            f"checkpoint_missing_keys path={path} count={len(missing)} example={missing[0]}"
        )
    if unexpected:
        log(
            f"checkpoint_unexpected_keys path={path} count={len(unexpected)} example={unexpected[0]}"
        )
    threshold_value = state.get("threshold") if isinstance(state, dict) else None
    if isinstance(threshold_value, (int, float)):
        return float(threshold_value)
    return None


def save_heatmaps(
    model: Fastflow,
    dataloader,
    scores_by_path: dict[str, float],
    threshold: float,
    result_cfg: ResultConfig,
    heatmap_cfg: HeatmapConfig,
) -> None:
    if not scores_by_path:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    root = result_cfg.root
    total = len(dataloader)
    for index, batch in enumerate(dataloader):
        images = batch.image.to(device)
        batch_paths_raw = (
            batch.image_path
            if isinstance(batch.image_path, list)
            else [batch.image_path]
        )
        with torch.no_grad():
            predictions = model(images)
            anomaly_map = getattr(predictions, "anomaly_map", None)
            if anomaly_map is None:
                inner: torch.nn.Module = getattr(model, "model", model)
                inner.eval()
                output = inner(images)
                if isinstance(output, tuple):
                    hidden_variables = output[0]
                    anomaly_map = inner.anomaly_map_generator(hidden_variables)
                else:
                    anomaly_map = output.anomaly_map
        if anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)
        maps_np = anomaly_map.detach().cpu().numpy()
        for path_raw, heat in zip(batch_paths_raw, maps_np):
            path_str = str(path_raw)
            score = scores_by_path.get(path_str)
            if score is None:
                continue
            src_path = Path(path_str)
            folder = (
                "good"
                if "good" in {p.lower() for p in src_path.parts}
                else "error"
                if "error" in {p.lower() for p in src_path.parts}
                else "unknown"
            )
            out_dir = root / "test" / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(src_path) as img:
                image = img.convert("RGB")
                if score <= threshold:
                    image.save(out_dir / src_path.name)
                    continue
                if heat.ndim == 3:
                    heat_2d = np.amax(heat, axis=0)
                else:
                    heat_2d = heat
                heat_2d = heat_2d.astype(np.float32)
                finite_mask = np.isfinite(heat_2d)
                if not finite_mask.any():
                    image.save(out_dir / src_path.name)
                    continue
                heat_2d = np.where(finite_mask, heat_2d, 0.0)
                heat_2d -= heat_2d.min()
                max_value = float(heat_2d.max())
                if max_value <= 0.0:
                    image.save(out_dir / src_path.name)
                    continue
                heat_norm = heat_2d / max_value
                percentile_cut = np.quantile(
                    heat_norm,
                    heatmap_cfg.activation_percentile
                    if 0.0 <= heatmap_cfg.activation_percentile <= 1.0
                    else 0.98,
                )
                dynamic_cut = max(heatmap_cfg.activation_min_value, percentile_cut)
                heat_mask = np.where(heat_norm >= dynamic_cut, heat_norm, 0.0)
                if heat_mask.max() <= 0:
                    image.save(out_dir / src_path.name)
                    continue
                heat_map_image = visualize_anomaly_map(
                    heat_mask, normalize=True, colormap=True
                )
                heat_map_image = heat_map_image.resize(
                    image.size, resample=Image.BILINEAR
                )
                overlay = Image.blend(image, heat_map_image, alpha=0.5)
                overlay.save(out_dir / src_path.name)
        log(f"heatmap_stage batch={index + 1}/{total}")


def main() -> None:
    seed_everything(42, workers=True, verbose=False)
    cfg, notes = load_config()
    log(f"config_model {asdict(cfg.model)}")
    log(f"config_data {asdict(cfg.data)}")
    log(f"config_output {asdict(cfg.output)}")
    log(f"config_infer {asdict(cfg.infer)}")
    log(f"config_result {asdict(cfg.result)}")
    for note in notes:
        log(f"config_note {note}")
    pre_processor = Fastflow.configure_pre_processor(
        image_size=(cfg.model.input_size, cfg.model.input_size)
    )
    train_transform, eval_transform = build_transforms(cfg, pre_processor)
    def prepare_datamodule(worker_count: int) -> Folder:
        return build_datamodule(
            cfg, train_transform, eval_transform, num_workers=worker_count
        )

    def log_dataset_stats(dm: Folder, worker_count: int) -> Tuple[int, int, int]:
        train_samples, val_samples, test_samples = summarize_datamodule(dm)
        log(
            "dataset_summary "
            f"train_samples={train_samples} "
            f"val_samples={val_samples} "
            f"test_samples={test_samples} "
            f"num_workers={worker_count}"
        )
        return train_samples, val_samples, test_samples

    requested_workers = cfg.data.num_workers
    datamodule = prepare_datamodule(requested_workers)
    train_dataset_size, val_dataset_size, test_dataset_size = log_dataset_stats(
        datamodule, requested_workers
    )
    model = build_model(cfg, pre_processor)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = cfg.output.model_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cfg.output.model_path
    loaded_threshold = load_checkpoint(checkpoint_path, model)
    trainer_kwargs = dict(
        max_epochs=cfg.model.num_epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        callbacks=[EpochLogger()],
        check_val_every_n_epoch=1,
    )

    def run_training(dm: Folder, worker_count: int) -> None:
        trainer = Trainer(**trainer_kwargs)
        log(f"train_start num_workers={worker_count}")
        trainer.fit(model=model, datamodule=dm)
        log("train_complete")

    if checkpoint_path.exists():
        log(f"train_skip checkpoint_found={checkpoint_path}")
        if loaded_threshold is not None:
            log(f"checkpoint_threshold_loaded value={loaded_threshold:.6f}")
    else:
        while True:
            try:
                run_training(datamodule, requested_workers)
                break
            except MisconfigurationException as exc:
                message = str(exc).lower()
                missing_metric = "conditioned on metric" in message
                if (
                    missing_metric
                    and cfg.model.lr_scheduler_monitor.lower() != "train_loss"
                ):
                    log(
                        "train_retry reason=missing_monitor "
                        "fallback_monitor=train_loss"
                    )
                    cfg.model.lr_scheduler_monitor = "train_loss"
                    apply_learning_rate(model, cfg.model)
                    continue
                raise
            except (PermissionError, RuntimeError) as exc:
                message = str(exc).lower()
                permission_issue = isinstance(exc, PermissionError) or any(
                    token in message for token in ("semlock", "please call `iter` first")
                )
                if requested_workers > 0 and permission_issue:
                    log(
                        "train_retry reason=multiprocessing_unavailable "
                        "fallback_num_workers=0"
                    )
                    requested_workers = 0
                    cfg.data.num_workers = 0
                    datamodule = prepare_datamodule(requested_workers)
                    train_dataset_size, val_dataset_size, test_dataset_size = (
                        log_dataset_stats(datamodule, requested_workers)
                    )
                    continue
                raise
    train_scores, train_labels, _ = compute_scores(
        model,
        datamodule.train_dataloader(),
        cfg.infer.pooling,
        cfg.infer.q,
        "train_scoring",
    )
    base_threshold = determine_threshold(train_scores, cfg.infer.epsilon)
    test_scores, test_labels, test_paths = compute_scores(
        model,
        datamodule.test_dataloader(),
        cfg.infer.pooling,
        cfg.infer.q,
        "test",
    )
    threshold = adapt_threshold(
        base_threshold, test_scores, test_labels, cfg.infer.epsilon
    )
    log(f"threshold_base={base_threshold:.6f}")
    log(f"threshold_final={threshold:.6f}")
    evaluate_predictions(test_scores, test_labels, test_paths, threshold)
    scores_by_path = {
        path: float(score) for path, score in zip(test_paths, test_scores)
    }
    save_heatmaps(
        model,
        datamodule.test_dataloader(),
        scores_by_path,
        threshold,
        cfg.result,
        cfg.heatmap,
    )
    save_checkpoint(cfg.output.model_path, model, threshold)


if __name__ == "__main__":
    try:
        print(sys.version)
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.exit(0)
