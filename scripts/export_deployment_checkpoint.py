"""Remove optimizer state from a SAM2 training snapshot for deployment."""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target.exists():
        raise FileExistsError(f"不会覆盖已有文件: {args.target}")
    snapshot = torch.load(args.source, map_location="cpu", weights_only=False)
    if not isinstance(snapshot, dict) or "model" not in snapshot:
        raise ValueError("训练 checkpoint 中未找到 model")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": snapshot["model"],
            "epoch": snapshot.get("epoch"),
            "best_meter_values": snapshot.get("best_meter_values", {}),
        },
        args.target,
    )
    print(args.target)


if __name__ == "__main__":
    main()
