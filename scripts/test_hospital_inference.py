"""Run one real SAM2 inference and verify the resulting case log."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_ui.app import _overlay  # noqa: E402
from web_ui.case_logger import CaseLogger  # noqa: E402
from web_ui.sam_backend import build_runner, ensure_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = ensure_rgb(Image.open(args.image))
    ground_truth = np.asarray(Image.open(args.mask)) > 0
    y_values, x_values = np.nonzero(ground_truth)
    if not len(x_values):
        raise ValueError("测试 mask 没有前景")
    point = (float(np.median(x_values)), float(np.median(y_values)), 1)
    runner = build_runner(str(args.checkpoint), backend="sam2", device=args.device)
    predictor = runner.new_predictor()
    runner.set_image(predictor, image)
    predicted_mask, score, _ = runner.predict(predictor, [point])
    logger = CaseLogger(
        args.output_dir,
        {
            "backend": runner.backend,
            "checkpoint": Path(runner.checkpoint).name,
            "model_type": runner.model_type,
            "input_size": runner.input_size,
            "config": runner.config,
        },
    )
    case_id = logger.start_case(image, "end-to-end-test")
    logger.record_interaction(
        case_id,
        "point",
        [point],
        mask=predicted_mask,
        score=score,
        point=point,
    )
    case_dir = logger.save_final(case_id, predicted_mask, _overlay(image, predicted_mask))
    manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if len(manifest["interactions"]) != 1 or not (case_dir / "final_mask.png").is_file():
        raise RuntimeError("端到端 case 日志校验失败")
    print(
        json.dumps(
            {
                "case_dir": str(case_dir),
                "score": score,
                "mask_pixels": int(predicted_mask.sum()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
