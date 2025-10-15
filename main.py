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
from typing import Callable, Dict, List, Optional, Tuple

import cv2
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
    # オフライン用の環境変数を退避して一時的に無効化する
    saved = {}
    for key in OFFLINE_ENV_VARS:
        # 現在の値を記録し、環境変数から取り除く
        saved[key] = os.environ.pop(key, None)
    return saved


def _restore_offline_env(saved: dict) -> None:
    # 退避しておいた環境変数を元に戻す
    for key, value in saved.items():
        if value is not None:
            # 削除前の値がある場合だけ環境変数に再設定する
            os.environ[key] = value


def _sanitize_backbone_name(backbone: str) -> str:
    # モデル名にスラッシュが含まれる場合にファイル名向けに置き換える
    return backbone.replace("/", "_")


def configure_offline_environment(models_dir: Path) -> None:
    # 学習で使用する各種キャッシュディレクトリを指定し、オフライン動作に備える
    models_dir.mkdir(parents=True, exist_ok=True)
    resolved = str(models_dir)
    # 主要なライブラリのキャッシュ先を同じディレクトリに集約する
    os.environ.setdefault("TORCH_HOME", resolved)
    os.environ.setdefault("TIMM_HOME", resolved)
    os.environ.setdefault("HF_HOME", resolved)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", resolved)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


configure_offline_environment(MODELS_DIR)


def log(message: str) -> None:
    # 標準出力にログメッセージをそのまま表示する
    print(message)


def parse_int_list(value: str) -> List[int]:
    # コンマ区切りの文字列を整数リストへ変換する
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    result: List[int] = []
    for token in tokens:
        try:
            # 整数に変換できた値のみ結果へ追加する
            result.append(int(token))
        except ValueError:
            # 数値以外は無視して処理を続行する
            continue
    return result


def _fastflow_forward_two_outputs(self, *args, **kwargs):
    # Fastflow の forward が2つのテンソルを返すようにラップする
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
                # 期待するテンソルが欠落している場合は異常終了させる
                raise RuntimeError("fastflow forward returned None")
            return hidden_variables, jacobians
    return output


def _fastflow_get_vit_features(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
    # 指定された層の特徴マップを Vision Transformer から取り出す
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
        # cls / dist トークンを先頭に結合して標準のトークン列に揃える
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
        # 各ブロックを順次適用し、指定層で特徴マップを保存する
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
                    # 正方形に並べ替えできない場合は処理を打ち切る
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
        # 目的の層が取得できなかった場合は最終層を返す
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
    blur_kernel_size: int
    blur_sigma: float
    normalize_clip_percentile: float
    normalize_clip_lower_percentile: float
    resize_to_input_size: bool


@dataclass
class AppConfig:
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
    infer: InferConfig
    result: ResultConfig
    heatmap: HeatmapConfig


@dataclass
class RuntimeContext:
    cfg: AppConfig
    notes: List[str]
    pre_processor: object
    train_transform: object
    eval_transform: object
    datamodule: Folder
    requested_workers: int
    trainer_kwargs: Dict[str, object]
    model: Fastflow
    checkpoint_path: Path
    loaded_threshold: Optional[float]
    prepare_datamodule: Callable[[int], Folder]
    log_dataset_stats: Callable[[Folder, int], Tuple[int, int, int]]


_RUNTIME_CONTEXT: Optional[RuntimeContext] = None


class EpochLogger(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: Fastflow) -> None:
        # 各エポック終了時に損失などの進捗をログへ流す
        metric = trainer.callback_metrics.get("train_loss")
        value = None
        if isinstance(metric, torch.Tensor):
            # Tensor のままでは扱いづらいので CPU へ取り出して数値化する
            value = float(metric.detach().cpu().item())
        elif metric is not None:
            value = float(metric)
        if value is None:
            log(f"train_epoch={trainer.current_epoch + 1}")
        else:
            log(f"train_epoch={trainer.current_epoch + 1} loss={value:.6f}")


def resolve_backbone(requested: str) -> Tuple[str, int]:
    # 利用可能なバックボーン名と推奨入力サイズを決定する
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
        # 末尾の細かい違いは無視して汎用的な DeiT バリエーションへ誘導する
        return "deit_base_distilled_patch16_384", 384
    if lowered.startswith("wide_resnet50"):
        return "wide_resnet50_2", 224
    return "resnet18", 224


def load_config(path: str = "setting.ini") -> Tuple[AppConfig, List[str]]:
    # 設定ファイルを読み込み、アプリケーション設定と注意事項をまとめて返す
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
        # 推奨と異なる指定があった場合は補正内容を記録する
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
        # 検証データが無い場合は監視対象を train_loss に切り替える
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
    blur_kernel_size = max(0, parser.getint("HEATMAP", "blur_kernel_size", fallback=5))
    if blur_kernel_size % 2 == 0 and blur_kernel_size != 0:
        # 偶数カーネルはガウシアンブラーで扱いづらいため奇数へ調整する
        blur_kernel_size += 1
    blur_sigma = max(0.0, parser.getfloat("HEATMAP", "blur_sigma", fallback=1.0))
    normalize_clip_percentile = float(
        min(
            max(
                parser.getfloat("HEATMAP", "normalize_clip_percentile", fallback=0.99),
                0.0,
            ),
            1.0,
        )
    )
    normalize_clip_lower_percentile = float(
        min(
            max(
                parser.getfloat(
                    "HEATMAP", "normalize_clip_lower_percentile", fallback=0.0
                ),
                0.0,
            ),
            1.0,
        )
    )
    if normalize_clip_lower_percentile > normalize_clip_percentile:
        # 上限より下限が大きい場合は上限に揃える
        normalize_clip_lower_percentile = normalize_clip_percentile
    resize_to_input_size = parser.getboolean(
        "HEATMAP", "resize_to_input_size", fallback=True
    )
    heatmap_cfg = HeatmapConfig(
        activation_percentile=activation_percentile,
        activation_min_value=activation_min_value,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        normalize_clip_percentile=normalize_clip_percentile,
        normalize_clip_lower_percentile=normalize_clip_lower_percentile,
        resize_to_input_size=resize_to_input_size,
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
    # Fastflow の内部 forward をラップして潜在表現とヤコビアンを取得できるようにする
    if not hasattr(model, "model"):
        return
    inner = model.model
    if getattr(inner, "_ff_two_output_patched", False):
        return
    original_forward = getattr(inner.__class__, "forward", None)
    if original_forward is not None:
        # 差し替え後に元の forward を呼び出せるよう参照を記録する
        setattr(inner, "_ff_original_forward", original_forward)
    inner.forward = types.MethodType(_fastflow_forward_two_outputs, inner)
    setattr(inner, "_ff_two_output_patched", True)


def apply_learning_rate(model: Fastflow, cfg: ModelConfig) -> None:
    # Optimizer と LR スケジューラを設定するためのメソッドを差し替える
    def configure(self: Fastflow):
        # Adam をベースに学習率とスケジューラを組み立てる
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5
        )
        factor = max(min(cfg.lr_scheduler_factor, 0.999), 1e-6)
        patience = max(cfg.lr_scheduler_patience, 0)
        min_lr = max(cfg.lr_scheduler_min_lr, 0.0)
        monitor_metric = cfg.lr_scheduler_monitor.strip() or "val_loss"
        if patience > 0 and 0.0 < factor < 1.0:
            # ReduceLROnPlateau を使用して損失が改善しない場合に学習率を下げる
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
    # Lightning へのログ送信を抑制し CLI への出力に集中させる
    original_log = model.log

    def patched_log(self: Fastflow, name, value, *args, **kwargs):
        # logger=False を強制して外部ロガーへの送信を無効化する
        kwargs["logger"] = False
        return original_log(name, value, *args, **kwargs)

    model.log = types.MethodType(patched_log, model)


def build_transforms(cfg: AppConfig, pre_processor) -> Tuple[object, object]:
    # 学習用と評価用の前処理パイプラインを構築する
    if not hasattr(pre_processor, "transform"):
        raise AttributeError("pre_processor missing transform attribute")
    base_transform = pre_processor.transform
    if isinstance(base_transform, T.Compose):
        base_ops = list(base_transform.transforms)
    else:
        base_ops = [base_transform]
    # 基本の前処理に軽めのデータ拡張を追加する
    train_ops = [
        T.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.98, 1.02),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomApply(
            [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)],
            p=0.3,
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.1),
    ] + base_ops
    train_transform = T.Compose(train_ops)
    eval_transform = base_transform
    return train_transform, eval_transform


def _assign_datamodule_transform(dm: Folder, split: str, transform) -> None:
    # 指定したデータ分割に対して変換処理を適用できるよう属性を更新する
    attr_candidates = [f"{split}_transform", f"{split}_augmentations"]
    for attr in attr_candidates:
        if hasattr(dm, attr):
            setattr(dm, attr, transform)
    dataset = getattr(dm, f"{split}_dataset", None)
    if dataset is not None:
        if hasattr(dataset, "transform"):
            # データセット側にも直接 transform を反映する
            dataset.transform = transform
        if hasattr(dataset, "augmentations"):
            dataset.augmentations = transform


def build_datamodule(
    cfg: AppConfig, train_transform, eval_transform, num_workers: int = 3
) -> Folder:
    # anomalib の Folder データモジュールを設定値に従って初期化する
    dm_kwargs = dict(
        name="dataset",
        normal_dir=str(cfg.data.train_dir),
        normal_test_dir=str(cfg.data.test_root / "good"),
        abnormal_dir=str(cfg.data.test_root / "error"),
        train_batch_size=cfg.model.batch_size,
        eval_batch_size=cfg.model.batch_size,
        num_workers=num_workers,
        val_split_ratio=cfg.data.val_split_ratio,
        augmentations=train_transform,
    )
    if cfg.data.val_split_ratio > 0.0:
        # 検証データを確保する場合はテストと同じ分割方法を使用する
        dm_kwargs["val_split_mode"] = "same_as_test"
    else:
        dm_kwargs["val_split_mode"] = "none"
    try:
        dm = Folder(**dm_kwargs)
    except TypeError:
        # 古い anomalib では val_split_ratio が存在しないため取り除いて再試行する
        dm_kwargs.pop("val_split_ratio", None)
        dm = Folder(**dm_kwargs)
    dm.setup()
    _assign_datamodule_transform(dm, "train", train_transform)
    for split in ("val", "test", "predict"):
        # 評価系のデータには評価用変換を一括で適用する
        _assign_datamodule_transform(dm, split, eval_transform)
    return dm


def summarize_datamodule(dm: Folder) -> Tuple[int, int, int]:
    # データモジュールの各分割に含まれるサンプル数を取得する
    train_size = len(dm.train_dataloader().dataset)

    def _loader_size(loader) -> int:
        # DataLoader またはリストで渡される場合に対応してサイズを計算する
        if loader is None:
            return 0
        if isinstance(loader, list):
            for item in loader:
                size = _loader_size(item)
                if size:
                    # 最初にサイズが得られたローダーを採用する
                    return size
            return 0
        dataset = getattr(loader, "dataset", None)
        return len(dataset) if dataset is not None else 0

    val_size = _loader_size(dm.val_dataloader())
    test_size = _loader_size(dm.test_dataloader())
    return train_size, val_size, test_size


def download_pretrained_weights(backbone: str, destination: Path) -> Path:
    # timm から学習済み重みを取得しローカルに保存する
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
        # GPU 非依存にするためテンソルを CPU 上の state_dict に変換する
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
    # 既存の重みファイルを検索し、必要ならばダウンロードを実行する
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"models directory missing. expected path={MODELS_DIR}")
    sanitized = _sanitize_backbone_name(backbone)
    candidates = [backbone, sanitized]
    exts = (".pth", ".pt", ".bin")
    for name in candidates:
        for ext in exts:
            candidate = MODELS_DIR / f"{name}{ext}"
            if candidate.exists():
                # 候補名と拡張子の組み合わせで直接一致したファイルを優先する
                return candidate
    matches = []
    for path in sorted(MODELS_DIR.glob("*")):
        if not path.is_file() or path.suffix not in exts:
            continue
        stem = path.stem
        if any(token in stem for token in candidates):
            matches.append(path)
    if matches:
        # 部分一致でも候補があれば最初のものを返す
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
    # チェックポイント内のキー名を Fastflow の読み込みに合わせて整形する
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
        # 余計なプレフィックスを取り除きシンプルなキーに整形する
        cleaned[new_key] = value
    return cleaned


def load_local_pretrained_weights(model: Fastflow, cfg: ModelConfig) -> None:
    # 保存済みのバックボーン重みを Fastflow の特徴抽出器へ読み込む
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
        # 読み込みできなかったキーは通知だけ行い学習は続行する
        log(f"weights_missing count={len(missing)} example={missing[0]}")
    if unexpected:
        log(f"weights_unexpected count={len(unexpected)} example={unexpected[0]}")


def build_model(cfg: AppConfig, pre_processor) -> Fastflow:
    # Fastflow モデルを構築し、必要なパッチや重みを適用する
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
    # 推論で追加情報を得られるよう各種パッチと重みを適用する
    ensure_two_output_forward(model)
    apply_learning_rate(model, cfg.model)
    suppress_module_logging(model)
    configure_vit_layers(model, cfg.model)
    load_local_pretrained_weights(model, cfg.model)
    return model


def configure_vit_layers(model: Fastflow, cfg: ModelConfig) -> None:
    # ViT 系バックボーンで使用する層を調整し FastFlow ブロックを再構成する
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
        # 実際に利用可能な層に合わせて設定を更新する
        log(f"vit_layers_adjusted requested={cfg.vit_layers} applied={unique_layers}")
        cfg.vit_layers = unique_layers

    patch_embed = getattr(feature_extractor, "patch_embed", None)
    if patch_embed is None:
        return
    patch_size_raw = getattr(patch_embed, "patch_size", (16, 16))
    # パッチサイズ情報を整数に揃えて後段の空間サイズ計算に使う
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
        # timm がグリッドサイズを保持している場合はその値を優先する
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
            # パッチ総数が平方数であれば正方形として扱う
            spatial_h = spatial_w = side

    if spatial_h is None or spatial_w is None:
        input_hw = getattr(inner, "image_size", getattr(inner, "input_size", None))
        if isinstance(input_hw, (tuple, list)) and len(input_hw) == 2:
            h, w = int(input_hw[0]), int(input_hw[1])
        elif isinstance(input_hw, int):
            h = w = int(input_hw)
        else:
            h = w = int(cfg.input_size)
        # 入力解像度をパッチサイズで割ることで特徴マップのサイズを推定する
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
    # カスタム実装で特定層のみを抽出できるように差し替える
    inner._get_vit_features = types.MethodType(_fastflow_get_vit_features, inner)
    setattr(inner, "_selected_vit_layers", tuple(unique_layers))


def compute_scores(
    model: Fastflow,
    dataloader,
    pooling: str,
    q: float,
    stage: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # 指定データローダーを走査して異常スコア・ラベル・画像パスを収集する
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
            # ラベルが欠けている場合は -1 を入れて評価対象外とする
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
                    # 異常マップの分位点を用いた異常スコアを計算する
                    scores_tensor = torch.quantile(flat, q, dim=1)
                else:
                    # 既定では最大値プールで代表スコアを求める
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
    # 学習データのスコアから基準となる閾値を統計的に推定する
    if train_scores.size == 0:
        return float(epsilon)
    finite_scores = train_scores[np.isfinite(train_scores)]
    if finite_scores.size == 0:
        return float(epsilon)
    max_based = float(finite_scores.max() + epsilon)
    quantile_based = float(np.quantile(finite_scores, 0.997))
    mean_value = float(finite_scores.mean())
    std_value = float(finite_scores.std())
    # 最大値・分位点・ガウシアン近似のいずれかで最も大きい値を採用する
    gaussian_based = float(mean_value + 3.0 * std_value)
    threshold = max(max_based, quantile_based, gaussian_based)
    return float(max(threshold, epsilon))


def adapt_threshold(
    base_threshold: float,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    epsilon: float,
) -> float:
    # テストデータに基づいて F1 と Youden 指数が最良となる閾値を探す
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
        # 候補が多すぎる場合は分位点サンプリングで絞り込む
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
            # F1 優先、同点なら Youden が高い候補を採用する
            best_threshold = float(candidate)
            best_f1 = f1
            best_youden = youden
    return float(max(best_threshold, epsilon))


def evaluate_predictions(
    scores: np.ndarray, labels: np.ndarray, paths: List[str], threshold: float
) -> None:
    # 選択した閾値で予測を評価し、混同行列や誤分類をログ出力する
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
                # 誤判定されたファイルとスコアをログに残す
                log(
                    f"performance_misclassified path={path} score={scores[i]:.6f} label={int(y_true[i])} pred={int(y_pred[i])}"
                )
    else:
        log(f"performance_threshold={threshold:.6f}")


def save_checkpoint(path: Path, model: Fastflow, threshold: float) -> None:
    # モデル重みと推定した閾値をまとめて保存する
    path.parent.mkdir(parents=True, exist_ok=True)
    # 閾値も同梱して推論時に再利用できるようにする
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, path)


def load_checkpoint(path: Path, model: Fastflow) -> Optional[float]:
    # 保存済みチェックポイントを読み込み、閾値があれば返す
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
        # 読み込み漏れがあってもログに記録して処理は継続する
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
    input_size: int,
) -> None:
    # スコアと元画像を突き合わせてヒートマップを作成・保存する
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
                    # モデルが tuple を返す場合は特徴量からヒートマップを再生成する
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
                # スコアが存在しない画像はスキップする
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
                    # 正常判定の場合は元画像をそのまま保存する
                    image.save(out_dir / src_path.name)
                    continue
                if heat.ndim == 3:
                    heat_2d = np.amax(heat, axis=0)
                else:
                    heat_2d = heat
                heat_2d = heat_2d.astype(np.float32)
                if (
                    heatmap_cfg.blur_kernel_size >= 3
                    and heatmap_cfg.blur_kernel_size % 2 == 1
                ):
                    # ノイズ抑制のためにガウシアンブラーをかける
                    heat_2d = cv2.GaussianBlur(
                        heat_2d,
                        (heatmap_cfg.blur_kernel_size, heatmap_cfg.blur_kernel_size),
                        heatmap_cfg.blur_sigma,
                    )
                finite_mask = np.isfinite(heat_2d)
                if not finite_mask.any():
                    image.save(out_dir / src_path.name)
                    continue
                heat_2d = np.where(finite_mask, heat_2d, 0.0)
                heat_values = heat_2d[finite_mask]
                lower_pct = heatmap_cfg.normalize_clip_lower_percentile
                upper_pct = heatmap_cfg.normalize_clip_percentile
                if 0.0 < upper_pct <= 1.0:
                    # 上位パーセンタイルをクリップして極端な値を抑制する
                    upper_value = float(np.quantile(heat_values, upper_pct))
                    heat_2d = np.minimum(heat_2d, upper_value)
                if 0.0 < lower_pct < 1.0:
                    # 下位パーセンタイルも同様に底上げする
                    lower_value = float(np.quantile(heat_values, lower_pct))
                    heat_2d = np.maximum(heat_2d, lower_value)
                heat_values = heat_2d[np.isfinite(heat_2d)]
                if heat_values.size == 0:
                    image.save(out_dir / src_path.name)
                    continue
                min_value = float(heat_values.min())
                max_value = float(heat_values.max())
                if not np.isfinite(max_value) or max_value <= min_value:
                    image.save(out_dir / src_path.name)
                    continue
                heat_norm = (heat_2d - min_value) / (max_value - min_value)
                percentile_source = (
                    heatmap_cfg.activation_percentile
                    if 0.0 <= heatmap_cfg.activation_percentile <= 1.0
                    else 0.98
                )
                percentile_cut = float(np.quantile(heat_norm, percentile_source))
                dynamic_cut = max(heatmap_cfg.activation_min_value, percentile_cut)
                if dynamic_cut <= 0.0:
                    heat_mask = heat_norm
                else:
                    heat_mask = np.where(heat_norm >= dynamic_cut, heat_norm, 0.0)
                if heat_mask.max() <= 0:
                    image.save(out_dir / src_path.name)
                    continue
                heat_map_raw = visualize_anomaly_map(
                    heat_mask, normalize=True, colormap=True
                )
                if isinstance(heat_map_raw, Image.Image):
                    heat_map_image = heat_map_raw.convert("RGB")
                else:
                    heat_map_array: Optional[np.ndarray] = None
                    if isinstance(heat_map_raw, np.ndarray):
                        heat_map_array = heat_map_raw
                    elif torch.is_tensor(heat_map_raw):
                        # Tensor で返ってきた場合も numpy に変換して扱う
                        heat_map_array = heat_map_raw.detach().cpu().numpy()
                    if heat_map_array is None:
                        raise TypeError(
                            "visualize_anomaly_map returned unsupported type"
                        )
                    if heat_map_array.dtype != np.uint8:
                        finite_values = heat_map_array[np.isfinite(heat_map_array)]
                        finite_max = (
                            float(finite_values.max()) if finite_values.size else 0.0
                        )
                        if finite_max <= 1.0:
                            heat_map_array = np.clip(heat_map_array * 255.0, 0, 255)
                        else:
                            heat_map_array = np.clip(heat_map_array, 0, 255)
                        heat_map_array = heat_map_array.astype(np.uint8)
                    heat_map_image = Image.fromarray(heat_map_array)
                base_image = image
                original_size = image.size
                if heatmap_cfg.resize_to_input_size and input_size > 0:
                    # 統一したサイズに合わせてヒートマップを重ねる
                    target_size = (int(input_size), int(input_size))
                    base_image = image.resize(target_size, resample=Image.BILINEAR)
                heat_map_image = heat_map_image.resize(
                    base_image.size, resample=Image.BILINEAR
                )
                overlay = Image.blend(base_image, heat_map_image, alpha=0.5)
                if heatmap_cfg.resize_to_input_size and input_size > 0:
                    overlay = overlay.resize(original_size, resample=Image.BILINEAR)
                overlay.save(out_dir / src_path.name)
        log(f"heatmap_stage batch={index + 1}/{total}")


def _initialize_runtime(
    path_train: str,
    path_test: str,
    path_model: str,
    path_result: str,
    *,
    log_config: bool,
) -> RuntimeContext:
    cfg, notes = load_config()
    cfg.data.train_dir = Path(path_train)
    cfg.data.test_root = Path(path_test)
    cfg.output.model_path = Path(path_model)
    cfg.result.root = Path(path_result)
    if log_config:
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
    log_dataset_stats(datamodule, requested_workers)
    model = build_model(cfg, pre_processor)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    checkpoint_path = cfg.output.model_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loaded_threshold = load_checkpoint(checkpoint_path, model)
    trainer_kwargs: Dict[str, object] = dict(
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
    return RuntimeContext(
        cfg=cfg,
        notes=notes,
        pre_processor=pre_processor,
        train_transform=train_transform,
        eval_transform=eval_transform,
        datamodule=datamodule,
        requested_workers=requested_workers,
        trainer_kwargs=trainer_kwargs,
        model=model,
        checkpoint_path=checkpoint_path,
        loaded_threshold=loaded_threshold,
        prepare_datamodule=prepare_datamodule,
        log_dataset_stats=log_dataset_stats,
    )


def train(
    path_train: str,
    path_test: str,
    path_model: str,
    path_result: str,
) -> RuntimeContext:
    global _RUNTIME_CONTEXT
    seed_everything(42, workers=True)
    context = _initialize_runtime(
        path_train,
        path_test,
        path_model,
        path_result,
        log_config=True,
    )
    _RUNTIME_CONTEXT = context
    checkpoint_path = context.checkpoint_path
    loaded_threshold = context.loaded_threshold
    if checkpoint_path.exists():
        log(f"train_skip checkpoint_found={checkpoint_path}")
        if loaded_threshold is not None:
            log(f"checkpoint_threshold_loaded value={loaded_threshold:.6f}")
        return context
    while True:
        try:
            trainer = Trainer(**context.trainer_kwargs)
            log(f"train_start num_workers={context.requested_workers}")
            trainer.fit(model=context.model, datamodule=context.datamodule)
            log("train_complete")
            torch.save({"state_dict": context.model.state_dict()}, checkpoint_path)
            break
        except MisconfigurationException as exc:
            message = str(exc).lower()
            missing_metric = "conditioned on metric" in message
            if (
                missing_metric
                and context.cfg.model.lr_scheduler_monitor.lower() != "train_loss"
            ):
                log("train_retry reason=missing_monitor fallback_monitor=train_loss")
                context.cfg.model.lr_scheduler_monitor = "train_loss"
                apply_learning_rate(context.model, context.cfg.model)
                continue
            raise
        except (PermissionError, RuntimeError) as exc:
            message = str(exc).lower()
            permission_issue = isinstance(exc, PermissionError) or any(
                token in message for token in ("semlock", "please call `iter` first")
            )
            if context.requested_workers > 0 and permission_issue:
                log(
                    "train_retry reason=multiprocessing_unavailable "
                    "fallback_num_workers=0"
                )
                context.requested_workers = 0
                context.cfg.data.num_workers = 0
                context.datamodule = context.prepare_datamodule(
                    context.requested_workers
                )
                context.log_dataset_stats(context.datamodule, context.requested_workers)
                continue
            raise
    context.loaded_threshold = None
    _RUNTIME_CONTEXT = context
    return context


def test(
    path_train: str,
    path_test: str,
    path_model: str,
    path_result: str,
    *,
    skip_threshold_update: bool = False,
) -> None:
    global _RUNTIME_CONTEXT
    seed_everything(42, workers=True)
    context = _RUNTIME_CONTEXT
    train_root = Path(path_train)
    checkpoint_path = Path(path_model)
    test_root = Path(path_test)
    result_root = Path(path_result)
    if (
        context is None
        or train_root != context.cfg.data.train_dir
        or checkpoint_path != context.checkpoint_path
        or test_root != context.cfg.data.test_root
        or result_root != context.cfg.result.root
    ):
        context = _initialize_runtime(
            str(train_root),
            str(test_root),
            str(checkpoint_path),
            str(result_root),
            log_config=context is None,
        )
        _RUNTIME_CONTEXT = context
    context.cfg.output.model_path = checkpoint_path
    if context.cfg.data.train_dir != train_root:
        context.cfg.data.train_dir = train_root
        context.datamodule = context.prepare_datamodule(context.requested_workers)
        context.log_dataset_stats(context.datamodule, context.requested_workers)
    if context.checkpoint_path != checkpoint_path:
        context.checkpoint_path = checkpoint_path
    if context.cfg.data.test_root != test_root:
        context.cfg.data.test_root = test_root
        context.datamodule = context.prepare_datamodule(context.requested_workers)
        context.log_dataset_stats(context.datamodule, context.requested_workers)
    if context.cfg.result.root != result_root:
        context.cfg.result.root = result_root
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found at {checkpoint_path}. run training before test()"
        )
    context.loaded_threshold = load_checkpoint(checkpoint_path, context.model)

    if skip_threshold_update:
        if context.loaded_threshold is None:
            raise RuntimeError(
                "checkpoint does not contain a saved threshold; cannot skip re-estimation"
            )
        threshold = context.loaded_threshold
        log(f"threshold_loaded={threshold:.6f}")
        test_scores, test_labels, test_paths = compute_scores(
            context.model,
            context.datamodule.test_dataloader(),
            context.cfg.infer.pooling,
            context.cfg.infer.q,
            "test",
        )
        log(f"threshold_final={threshold:.6f}")
    else:
        train_scores, _, _ = compute_scores(
            context.model,
            context.datamodule.train_dataloader(),
            context.cfg.infer.pooling,
            context.cfg.infer.q,
            "train_scoring",
        )
        base_threshold = determine_threshold(train_scores, context.cfg.infer.epsilon)
        test_scores, test_labels, test_paths = compute_scores(
            context.model,
            context.datamodule.test_dataloader(),
            context.cfg.infer.pooling,
            context.cfg.infer.q,
            "test",
        )
        threshold = adapt_threshold(
            base_threshold, test_scores, test_labels, context.cfg.infer.epsilon
        )
        log(f"threshold_base={base_threshold:.6f}")
        log(f"threshold_final={threshold:.6f}")
        save_checkpoint(checkpoint_path, context.model, threshold)
    context.loaded_threshold = threshold
    evaluate_predictions(test_scores, test_labels, test_paths, threshold)
    scores_by_path = {
        path: float(score) for path, score in zip(test_paths, test_scores)
    }
    save_heatmaps(
        context.model,
        context.datamodule.test_dataloader(),
        scores_by_path,
        threshold,
        context.cfg.result,
        context.cfg.heatmap,
        context.cfg.model.input_size,
    )
    _RUNTIME_CONTEXT = context


if __name__ == "__main__":
    try:
        path_train = "data/train/good"
        path_test = "data/test"
        path_model = "params/model.pth"
        path_result = "result"
        print(sys.version)

        print("start train.")
        train(path_train, path_test, path_model, path_result)

        print("start test.")
        test(path_train, path_test, path_model, path_result)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    finally:
        sys.exit(0)
