"""Gradio web interface for click-prompt segmentation with SAM1/SAM2 weights."""

import argparse
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
from PIL import Image

try:
    from .sam_backend import Point, SessionData, SessionStore, build_runner, ensure_rgb
except ImportError:
    from sam_backend import Point, SessionData, SessionStore, build_runner, ensure_rgb


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / (
    "external/sam2/sam2_logs/lung_hiera_l_1024_f1_bs7_12ep_full_valbest_v2/"
    "checkpoints/val_all_seg_slice_iou_mean.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2/sam2_hiera_l.yaml"
IMAGE_FIT_CSS = (
    ".sam-full-image img { object-fit: contain !important; }"
    ".sam-auto-height img { height: auto !important; }"
)


def _session_id(request: Optional[gr.Request]) -> str:
    session_hash = getattr(request, "session_hash", None)
    return str(session_hash or "local")


def _event_xy(event: Any) -> Tuple[float, float]:
    index = getattr(event, "index", None)
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise ValueError("无法读取点击坐标，请重新点击图片")
    return float(index[0]), float(index[1])


def _draw_points(image: np.ndarray, points: Sequence[Point]) -> np.ndarray:
    output = image.copy()
    radius = max(4, min(output.shape[:2]) // 100)
    for x_value, y_value, label in points:
        x, y = int(round(x_value)), int(round(y_value))
        color = np.array([0, 210, 80] if label == 1 else [235, 60, 60], dtype=np.uint8)
        y0, y1 = max(0, y - radius), min(output.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(output.shape[1], x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        output[y0:y1, x0:x1][circle] = color
    return output


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32).copy()
    color = np.array([35, 145, 255], dtype=np.float32)
    foreground = mask.astype(bool)
    output[foreground] = output[foreground] * 0.55 + color * 0.45
    return np.clip(output, 0, 255).astype(np.uint8)


def _points_json(points: Sequence[Point]) -> List[dict]:
    return [
        {
            "序号": index,
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "类型": "前景" if label else "背景",
        }
        for index, (x, y, label) in enumerate(points, start=1)
    ]


def _render(data: SessionData, status: Optional[str] = None):
    if data.image is None:
        return None, None, _points_json(data.points), status or "请先上传一张 2D 图片。"
    display = _draw_points(data.image, data.points)
    if data.mask is None:
        default_status = "图片已加载，请选择点击类型后点击目标。"
        return display, None, _points_json(data.points), status or default_status
    display = _draw_points(_overlay(data.image, data.mask), data.points)
    mask_image = (data.mask.astype(np.uint8) * 255)
    score_text = f"分割完成，最佳 mask score：{data.score:.4f}" if data.score is not None else "分割完成。"
    return display, mask_image, _points_json(data.points), status or score_text


def build_demo(runner: Any, output_dir: str = "web_outputs") -> gr.Blocks:
    store = SessionStore(runner)
    output_root = Path(output_dir).expanduser().resolve()

    def upload_image(image, request: gr.Request):
        session = store.get(_session_id(request))
        if image is None:
            store.reset(_session_id(request))
            return _render(session, "等待上传图片。")
        try:
            rgb = ensure_rgb(image)
            with session.lock:
                runner.set_image(session.predictor, rgb)
                session.image = rgb
                session.points.clear()
                session.mask = None
                session.score = None
                session.low_res_mask = None
            return _render(session, f"图片已加载（{rgb.shape[1]}×{rgb.shape[0]}），正在等待点击。")
        except Exception as exc:
            store.reset(_session_id(request))
            return _render(session, f"图片加载失败：{exc}")

    def select_point(mode: str, request: gr.Request, event: gr.SelectData):
        session = store.get(_session_id(request))
        if session.image is None:
            return _render(session, "请先上传图片。")
        try:
            x, y = _event_xy(event)
            height, width = session.image.shape[:2]
            x = max(0.0, min(float(width - 1), x))
            y = max(0.0, min(float(height - 1), y))
            label = 0 if mode == "背景点" else 1
            with session.lock:
                session.points.append((x, y, label))
                session.mask, session.score, session.low_res_mask = runner.predict(
                    session.predictor,
                    session.points,
                    mask_input=session.low_res_mask,
                )
            return _render(session)
        except Exception as exc:
            return _render(session, f"点击分割失败：{exc}")

    def undo_point(request: gr.Request):
        session = store.get(_session_id(request))
        with session.lock:
            if session.points:
                session.points.pop()
            if session.points:
                session.mask, session.score, session.low_res_mask = runner.predict(
                    session.predictor, session.points
                )
            else:
                session.mask, session.score, session.low_res_mask = None, None, None
        message = "已撤销最后一个点击。" if session.points or session.image is not None else None
        return _render(session, message)

    def clear_points(request: gr.Request):
        session = store.get(_session_id(request))
        with session.lock:
            session.points.clear()
            session.mask, session.score, session.low_res_mask = None, None, None
        return _render(session, "已清空 prompt。")

    def reset_image(request: gr.Request):
        session = store.reset(_session_id(request))
        return _render(session, "已重置当前会话。")

    def save_results(request: gr.Request):
        session = store.get(_session_id(request))
        if session.image is None or session.mask is None:
            return "请先上传图片并点击目标生成 mask。"
        safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", _session_id(request))
        session_dir = output_root / safe_session
        session_dir.mkdir(parents=True, exist_ok=True)
        mask_path = session_dir / "mask.png"
        overlay_path = session_dir / "overlay.png"
        with session.lock:
            Image.fromarray((session.mask.astype(np.uint8) * 255), mode="L").save(mask_path)
            Image.fromarray(_overlay(session.image, session.mask), mode="RGB").save(overlay_path)
        return f"结果已保存到 `{session_dir}`。"

    with gr.Blocks(title="SAM Interactive Segmentation", css=IMAGE_FIT_CSS) as demo:
        gr.Markdown(
            """# SAM 交互式分割
上传原图后选择点击类型并点击目标。绿色为前景点，红色为背景点；每次点击都会更新分割结果。"""
        )
        with gr.Row():
            with gr.Column(scale=3):
                image_input = gr.Image(
                    label="原图与预测叠加（上传后点击）",
                    type="numpy",
                    height=760,
                    interactive=True,
                    elem_classes=["sam-full-image"],
                )
                click_mode = gr.Radio(["前景点", "背景点"], value="前景点", label="点击类型")
                with gr.Row():
                    undo_button = gr.Button("撤销最后一点")
                    clear_button = gr.Button("清空 prompt")
                    reset_button = gr.Button("重置图片")
                status = gr.Markdown("等待上传图片。")
                points = gr.JSON(label="当前 prompt", value=[])
            with gr.Column(scale=2):
                mask_output = gr.Image(
                    label="二值 mask",
                    type="numpy",
                    image_mode="L",
                    format="png",
                    elem_classes=["sam-full-image", "sam-auto-height"],
                )
                save_button = gr.Button("保存结果", variant="primary")

        render_outputs = [
            image_input,
            mask_output,
            points,
            status,
        ]
        image_input.upload(upload_image, inputs=[image_input], outputs=render_outputs)
        image_input.select(select_point, inputs=[click_mode], outputs=render_outputs)
        undo_button.click(undo_point, inputs=[], outputs=render_outputs)
        clear_button.click(clear_points, inputs=[], outputs=render_outputs)
        reset_button.click(reset_image, inputs=[], outputs=render_outputs)
        save_button.click(save_results, inputs=[], outputs=[status])
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM click-prompt Gradio web UI")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SAM_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
        help="SAM1/SAM2 checkpoint；默认使用肺部 SAM2 prompt-trained 权重",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("SAM_BACKEND", "auto"),
        choices=("auto", "sam1", "sam2"),
        help="推理后端；auto 会根据默认 checkpoint 自动选择",
    )
    parser.add_argument(
        "--model-type",
        default=os.environ.get("SAM_MODEL_TYPE", "vit_b"),
        choices=sorted({"vit_b", "vit_l", "vit_h", "default"}),
    )
    parser.add_argument("--device", default=os.environ.get("SAM_DEVICE", "auto"))
    parser.add_argument(
        "--input-size",
        type=int,
        default=int(os.environ.get("SAM_INPUT_SIZE", "0")),
        help="SAM1 训练时的输入边长；0 表示从 checkpoint 自动推断，SAM2 使用配置值",
    )
    parser.add_argument(
        "--sam2-config",
        default=os.environ.get("SAM2_CONFIG", DEFAULT_SAM2_CONFIG),
        help="SAM2 Hydra 配置路径或相对于 external/sam2/sam2 的配置名",
    )
    parser.add_argument("--output-dir", default=os.environ.get("SAM_WEB_OUTPUT_DIR", "web_outputs"))
    parser.add_argument("--server-name", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
    parser.add_argument("--share", action="store_true", help="生成临时公网分享链接")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = build_runner(
        checkpoint=args.checkpoint,
        backend=args.backend,
        model_type=args.model_type,
        sam2_config=args.sam2_config,
        device=args.device,
        input_size=args.input_size,
    )
    print(f"Backend: {runner.backend}")
    print(f"Checkpoint: {runner.checkpoint}")
    if runner.backend == "sam2":
        print(f"Model: SAM2 {runner.model_type}, config={runner.config}")
    else:
        print(f"Model: SAM1 {runner.model_type}")
    print(f"Input size: {runner.input_size}, device={runner.device}")
    demo = build_demo(runner, args.output_dir)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
