"""Gradio web interface for click-prompt segmentation with custom SAM1 weights."""

import argparse
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
from PIL import Image

try:
    from .sam_backend import Point, SAMRunner, SessionData, SessionStore, ensure_rgb
except ImportError:
    from sam_backend import Point, SAMRunner, SessionData, SessionStore, ensure_rgb


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
        return None, None, None, _points_json(data.points), status or "请先上传一张 2D 图片。", None, None
    display = _draw_points(data.image, data.points)
    if data.mask is None:
        default_status = "图片已加载，请选择点击类型后点击目标。"
        return display, None, None, _points_json(data.points), status or default_status, None, None
    overlay = _overlay(data.image, data.mask)
    mask_image = (data.mask.astype(np.uint8) * 255)
    score_text = f"分割完成，最佳 mask score：{data.score:.4f}" if data.score is not None else "分割完成。"
    return display, overlay, mask_image, _points_json(data.points), status or score_text, None, None


def build_demo(runner: SAMRunner, output_dir: str = "web_outputs") -> gr.Blocks:
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
                session.predictor.set_image(rgb)
                session.image = rgb
                session.points.clear()
                session.mask = None
                session.score = None
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
                session.mask, session.score = runner.predict(session.predictor, session.points)
            return _render(session)
        except Exception as exc:
            return _render(session, f"点击分割失败：{exc}")

    def undo_point(request: gr.Request):
        session = store.get(_session_id(request))
        with session.lock:
            if session.points:
                session.points.pop()
            if session.points:
                session.mask, session.score = runner.predict(session.predictor, session.points)
            else:
                session.mask, session.score = None, None
        message = "已撤销最后一个点击。" if session.points or session.image is not None else None
        return _render(session, message)

    def clear_points(request: gr.Request):
        session = store.get(_session_id(request))
        with session.lock:
            session.points.clear()
            session.mask, session.score = None, None
        return _render(session, "已清空 prompt。")

    def reset_image(request: gr.Request):
        session = store.reset(_session_id(request))
        return _render(session, "已重置当前会话。")

    def save_results(request: gr.Request):
        session = store.get(_session_id(request))
        if session.image is None or session.mask is None:
            return None, None, "请先上传图片并点击目标生成 mask。"
        safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", _session_id(request))
        session_dir = output_root / safe_session
        session_dir.mkdir(parents=True, exist_ok=True)
        mask_path = session_dir / "mask.png"
        overlay_path = session_dir / "overlay.png"
        with session.lock:
            Image.fromarray((session.mask.astype(np.uint8) * 255), mode="L").save(mask_path)
            Image.fromarray(_overlay(session.image, session.mask), mode="RGB").save(overlay_path)
        return str(mask_path), str(overlay_path), f"结果已保存到 `{session_dir}`。"

    with gr.Blocks(title="Custom SAM1 Interactive Segmentation") as demo:
        gr.Markdown(
            """# Custom SAM1 交互式分割
上传原图后选择点击类型并点击目标。绿色为前景点，红色为背景点；每次点击都会更新分割结果。"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="原图（上传后点击）",
                    type="numpy",
                    height=600,
                    interactive=True,
                )
                click_mode = gr.Radio(["前景点", "背景点"], value="前景点", label="点击类型")
                with gr.Row():
                    undo_button = gr.Button("撤销最后一点")
                    clear_button = gr.Button("清空 prompt")
                    reset_button = gr.Button("重置图片")
                status = gr.Markdown("等待上传图片。")
                points = gr.JSON(label="当前 prompt", value=[])
            with gr.Column(scale=1):
                overlay_output = gr.Image(label="分割叠加结果", type="numpy", height=450)
                mask_output = gr.Image(label="二值 mask", type="numpy", image_mode="L", height=450)
                save_button = gr.Button("保存并下载结果", variant="primary")
                mask_file = gr.File(label="mask PNG", file_count="single")
                overlay_file = gr.File(label="overlay PNG", file_count="single")

        render_outputs = [
            image_input,
            overlay_output,
            mask_output,
            points,
            status,
            mask_file,
            overlay_file,
        ]
        image_input.upload(upload_image, inputs=[image_input], outputs=render_outputs)
        image_input.select(select_point, inputs=[click_mode], outputs=render_outputs)
        undo_button.click(undo_point, inputs=[], outputs=render_outputs)
        clear_button.click(clear_points, inputs=[], outputs=render_outputs)
        reset_button.click(reset_image, inputs=[], outputs=render_outputs)
        save_button.click(save_results, inputs=[], outputs=[mask_file, overlay_file, status])
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom SAM1 click-prompt Gradio web UI")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SAM_CHECKPOINT", "checkpoints/sam_vit_b_01ec64.pth"),
        help="SAM1 原始 checkpoint 或包含 model_state_dict 的训练快照",
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
        help="训练时的输入边长；0 表示从 checkpoint 自动推断",
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
    runner = SAMRunner(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        input_size=args.input_size,
    )
    print(f"Checkpoint: {runner.checkpoint}")
    print(f"Model: sam_{runner.model_type}, input_size={runner.input_size}, device={runner.device}")
    demo = build_demo(runner, args.output_dir)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
