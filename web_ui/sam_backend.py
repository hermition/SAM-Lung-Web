"""Backend helpers for the SAM1/SAM2 interactive web UI."""

from __future__ import annotations

import contextlib
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from segment_anything import SamPredictor, sam_model_registry


Point = Tuple[float, float, int]


def ensure_rgb(image: object) -> np.ndarray:
    """Convert a PIL/numpy image to contiguous uint8 RGB data."""
    if isinstance(image, Image.Image):
        array = np.asarray(image)
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("图片必须是 HxW、HxWx1、HxWx3 或 HxWx4 格式")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _unwrap_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint 必须是 state dict 或训练快照字典")

    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError("checkpoint 中未找到 model_state_dict/state_dict")

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            continue
        normalized[key.removeprefix("module.")] = value
    return normalized


def _infer_input_size(state_dict: Dict[str, torch.Tensor], patch_size: int) -> Optional[int]:
    pos_embed = state_dict.get("image_encoder.pos_embed")
    if pos_embed is None or not hasattr(pos_embed, "shape") or len(pos_embed.shape) != 4:
        return None
    embedding_size = int(pos_embed.shape[1])
    inferred = embedding_size * patch_size
    return inferred if inferred > 0 else None


def set_sam_input_size(model: torch.nn.Module, input_size: int) -> None:
    """Match the SAM image/prompt encoder shapes used during fine-tuning."""
    current_size = int(model.image_encoder.img_size)
    if input_size == current_size:
        return
    patch_size = int(model.image_encoder.patch_embed.proj.kernel_size[0])
    if input_size <= 0 or input_size % patch_size != 0:
        raise ValueError(f"input_size 必须是 {patch_size} 的正整数倍，得到 {input_size}")

    embedding_size = input_size // patch_size
    pos_embed = model.image_encoder.pos_embed
    if pos_embed is not None:
        resized = F.interpolate(
            pos_embed.permute(0, 3, 1, 2),
            size=(embedding_size, embedding_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        model.image_encoder.pos_embed = torch.nn.Parameter(resized)

    model.image_encoder.img_size = input_size
    model.prompt_encoder.input_image_size = (input_size, input_size)
    model.prompt_encoder.image_embedding_size = (embedding_size, embedding_size)
    model.prompt_encoder.mask_input_size = (4 * embedding_size, 4 * embedding_size)


class SAMRunner:
    """Load one custom SAM1 checkpoint and run image/prompt inference."""

    def __init__(
        self,
        checkpoint: str,
        model_type: str,
        device: str = "auto",
        input_size: int = 0,
    ) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
        if model_type not in sam_model_registry:
            choices = ", ".join(sorted(sam_model_registry))
            raise ValueError(f"不支持的 model_type={model_type}，可选值: {choices}")

        requested_device = device.lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU")

        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _unwrap_state_dict(payload)
        base_model = sam_model_registry[model_type](checkpoint=None)
        patch_size = int(base_model.image_encoder.patch_embed.proj.kernel_size[0])
        resolved_input_size = input_size or _infer_input_size(state_dict, patch_size) or 1024
        set_sam_input_size(base_model, int(resolved_input_size))
        try:
            base_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "checkpoint 与 SAM1 模型结构或 input_size 不匹配；"
                "请确认 --model-type、--input-size 与训练配置一致。"
            ) from exc

        self.checkpoint = str(checkpoint_path)
        self.model_type = model_type
        self.input_size = int(resolved_input_size)
        self.device = torch.device(requested_device)
        self.backend = "sam1"
        self._inference_lock = threading.Lock()
        self.model = base_model.to(self.device).eval()

    def new_predictor(self) -> SamPredictor:
        return SamPredictor(self.model)

    def set_image(self, predictor: SamPredictor, image: np.ndarray) -> None:
        with self._inference_lock, torch.inference_mode():
            predictor.set_image(image)

    @torch.no_grad()
    def predict(
        self,
        predictor: SamPredictor,
        points: Sequence[Point],
        mask_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        if not points:
            raise ValueError("至少需要一个点击点")
        point_coords = np.asarray([(point[0], point[1]) for point in points], dtype=np.float32)
        point_labels = np.asarray([point[2] for point in points], dtype=np.int32)
        with self._inference_lock:
            masks, scores, low_res_masks = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=len(points) == 1,
            )
        best_index = int(np.argmax(scores))
        low_res = np.asarray(low_res_masks[best_index], dtype=np.float32)
        if low_res.ndim == 2:
            low_res = low_res[None, ...]
        return masks[best_index].astype(bool), float(scores[best_index]), low_res


def _sam2_source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "external" / "sam2"


def _sam2_config_name(config: str, source_root: Path) -> str:
    config_path = Path(config).expanduser()
    if not config_path.is_absolute():
        local_path = (Path.cwd() / config_path).resolve()
        config_root = (source_root / "sam2").resolve()
        if local_path.is_file():
            try:
                return local_path.relative_to(config_root).as_posix()
            except ValueError:
                pass
        return str(config).replace("\\", "/")
    config_root = source_root / "sam2"
    try:
        return config_path.resolve().relative_to(config_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"SAM2 配置必须位于 {config_root} 下，得到 {config_path}"
        ) from exc


def _load_sam2_api(source_root: Path):
    if not source_root.is_dir():
        raise FileNotFoundError(f"找不到 SAM2 源码目录: {source_root}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM2 依赖未安装，请确认环境中有 hydra-core、iopath，并保留 external/sam2 源码。"
        ) from exc
    return build_sam2, SAM2ImagePredictor


class SAM2Runner:
    """Load a fine-tuned SAM2 checkpoint and run static-image prompt inference."""

    def __init__(
        self,
        checkpoint: str,
        config: str = "configs/sam2/sam2_hiera_l.yaml",
        device: str = "auto",
    ) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

        requested_device = device.lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU")

        source_root = _sam2_source_root()
        build_sam2, predictor_class = _load_sam2_api(source_root)
        config_name = _sam2_config_name(config, source_root)
        resolved_device = torch.device(requested_device)
        model = build_sam2(
            config_name,
            str(checkpoint_path),
            device=str(resolved_device),
            mode="eval",
            apply_postprocessing=False,
        )

        self.checkpoint = str(checkpoint_path)
        self.config = config_name
        self.model_type = "hiera_l"
        self.input_size = int(model.image_size)
        self.device = resolved_device
        self.backend = "sam2"
        self._predictor_class = predictor_class
        self._inference_lock = threading.Lock()
        self.model = model.eval()

    def new_predictor(self):
        return self._predictor_class(self.model)

    def _autocast_context(self):
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        try:
            supports_bf16 = torch.cuda.is_bf16_supported()
        except AttributeError:
            supports_bf16 = False
        dtype = torch.bfloat16 if supports_bf16 else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def set_image(self, predictor: Any, image: np.ndarray) -> None:
        with self._inference_lock, torch.inference_mode(), self._autocast_context():
            predictor.set_image(image)

    def predict(
        self,
        predictor: Any,
        points: Sequence[Point],
        mask_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        if not points:
            raise ValueError("至少需要一个点击点")
        point_coords = np.asarray([(point[0], point[1]) for point in points], dtype=np.float32)
        point_labels = np.asarray([point[2] for point in points], dtype=np.int32)
        with self._inference_lock, torch.inference_mode(), self._autocast_context():
            masks, scores, low_res_masks = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=len(points) == 1,
                return_logits=False,
                # Points from the web UI are in original-image pixel coordinates.
                normalize_coords=True,
            )
        best_index = int(np.argmax(scores))
        low_res = np.asarray(low_res_masks[best_index], dtype=np.float32)
        if low_res.ndim == 2:
            low_res = low_res[None, ...]
        return masks[best_index].astype(bool), float(scores[best_index]), low_res


def build_runner(
    checkpoint: str,
    backend: str = "auto",
    model_type: str = "vit_b",
    sam2_config: str = "configs/sam2/sam2_hiera_l.yaml",
    device: str = "auto",
    input_size: int = 0,
):
    """Build a SAM1/SAM2 runner while keeping the old SAM1 entry point usable."""
    selected_backend = backend.lower()
    if selected_backend == "auto":
        checkpoint_text = str(Path(checkpoint).expanduser()).lower().replace("\\", "/")
        selected_backend = (
            "sam2"
            if "external/sam2" in checkpoint_text
            or "sam2" in Path(checkpoint).name.lower()
            or Path(checkpoint).name == "val_all_seg_slice_iou_mean.pt"
            else "sam1"
        )
    if selected_backend == "sam2":
        return SAM2Runner(checkpoint=checkpoint, config=sam2_config, device=device)
    if selected_backend == "sam1":
        return SAMRunner(
            checkpoint=checkpoint,
            model_type=model_type,
            device=device,
            input_size=input_size,
        )
    raise ValueError(f"不支持的 backend={backend}，可选值: auto、sam1、sam2")


@dataclass
class SessionData:
    predictor: Any
    image: Optional[np.ndarray] = None
    points: List[Point] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    score: Optional[float] = None
    low_res_mask: Optional[np.ndarray] = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class SessionStore:
    """Keep image embeddings isolated between browser sessions."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(self.runner.new_predictor())
            return self._sessions[session_id]

    def reset(self, session_id: str) -> SessionData:
        data = self.get(session_id)
        with data.lock:
            reset_predictor = getattr(data.predictor, "reset_image", None)
            if reset_predictor is None:
                reset_predictor = getattr(data.predictor, "reset_predictor", None)
            if reset_predictor is not None:
                reset_predictor()
            data.image = None
            data.points.clear()
            data.mask = None
            data.score = None
            data.low_res_mask = None
        return data
