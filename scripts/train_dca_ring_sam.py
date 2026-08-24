#!/usr/bin/env python
import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment_anything import sam_model_registry

DEFAULT_CHECKPOINTS = {
    "vit_b": REPO_ROOT / "checkpoints" / "sam_vit_b_01ec64.pth",
    "vit_l": REPO_ROOT / "checkpoints" / "sam_vit_l_0b3195.pth",
    "vit_h": REPO_ROOT / "checkpoints" / "sam_vit_h_4b8939.pth",
}
SELECTION_METRIC = "val_dice_mean"
DEFAULT_DCA_RING_DATASET_ROOT = "/data_new/moyancheng/dataset/dca_dataset_ring"
DEFAULT_LUNG_SPLIT_ROOT = (
    "/data_new/moyancheng/dataset/lung/肺癌勾画/processed_axial_slices_png_0mm/data_spilt"
)


@dataclass
class Sample:
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM mask decoder on dca_dataset_ring without prompts."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DCA_RING_DATASET_ROOT,
    )
    parser.add_argument(
        "--dataset-format",
        choices=("dir_splits", "manifest_splits"),
        default="dir_splits",
    )
    parser.add_argument(
        "--model-type",
        choices=tuple(DEFAULT_CHECKPOINTS.keys()),
        default="vit_b",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default="sam-dca-ring")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--train-mode",
        choices=("decoder_only", "full"),
        default="decoder_only",
        help="Train only the mask decoder or jointly train image encoder and decoder.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the requested device.")
        torch.cuda.set_device(resolved)
    elif resolved.type != "cpu":
        raise RuntimeError(f"Unsupported device: {device}")
    return resolved


def resolve_checkpoint_path(model_type: str, checkpoint: Optional[str]) -> Path:
    if checkpoint is not None:
        return Path(checkpoint).resolve()
    return DEFAULT_CHECKPOINTS[model_type].resolve()


def dataset_tag(dataset_format: str) -> str:
    if dataset_format == "manifest_splits":
        return "lung"
    return "dca_ring"


def load_samples(dataset_root: Path, split: str, limit: Optional[int]) -> List[Sample]:
    image_dir = dataset_root / split / "imgs"
    mask_dir = dataset_root / split / "masks" / "0"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image dir: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

    image_files = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    samples: List[Sample] = []
    for image_path in image_files:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        samples.append(Sample(image_path=image_path, mask_path=mask_path))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def pil_resize_image(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=Image.BILINEAR)


def pil_resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), resample=Image.NEAREST)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


class DCARingTrainDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], input_size: int) -> None:
        self.samples = list(samples)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")
        image = pil_resize_image(image, self.input_size)
        mask = pil_resize_mask(mask, self.input_size)
        image_arr = np.asarray(image, dtype=np.float32)
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).contiguous()
        mask_tensor = mask_to_tensor(mask)
        return image_tensor, mask_tensor


def load_samples_from_manifest(manifest_path: Path, limit: Optional[int]) -> List[Sample]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    samples: List[Sample] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_dir = Path(row["image_path"])
            mask_dir = Path(row["mask_path"])
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Missing image dir: {image_dir}")
            if not mask_dir.is_dir():
                raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

            image_files = sorted(
                p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for image_path in image_files:
                mask_path = mask_dir / image_path.name
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
                samples.append(Sample(image_path=image_path, mask_path=mask_path))
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples


def load_split_samples(
    dataset_root: Path,
    dataset_format: str,
    split: str,
    limit: Optional[int],
) -> List[Sample]:
    if dataset_format == "dir_splits":
        return load_samples(dataset_root, split, limit)
    manifest_path = dataset_root / f"case_report_manifest_{split}.csv"
    return load_samples_from_manifest(manifest_path, limit)


def set_sam_input_size(model: nn.Module, input_size: int) -> None:
    current_size = model.image_encoder.img_size
    if input_size == current_size:
        return
    patch_size = model.image_encoder.patch_embed.proj.kernel_size[0]
    if input_size % patch_size != 0:
        raise ValueError(f"--input-size must be divisible by {patch_size}, got {input_size}")
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


def configure_trainable_params(model: nn.Module, train_mode: str) -> None:
    for param in model.image_encoder.parameters():
        param.requires_grad = train_mode == "full"
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = True


def preprocess_batch(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return torch.stack([model.preprocess(image) for image in images], dim=0)


def forward_no_prompt(
    model: nn.Module,
    images: torch.Tensor,
    freeze_image_encoder: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_images = preprocess_batch(model, images)
    if freeze_image_encoder:
        with torch.no_grad():
            image_embeddings = model.image_encoder(input_images)
    else:
        image_embeddings = model.image_encoder(input_images)
    low_res_masks_list: List[torch.Tensor] = []
    iou_predictions_list: List[torch.Tensor] = []
    image_pe = model.prompt_encoder.get_dense_pe()
    dense_embeddings = model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        1,
        -1,
        model.prompt_encoder.image_embedding_size[0],
        model.prompt_encoder.image_embedding_size[1],
    )
    sparse_embeddings = torch.empty(
        (1, 0, model.prompt_encoder.embed_dim),
        device=images.device,
    )
    for curr_embedding in image_embeddings:
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=curr_embedding.unsqueeze(0),
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        low_res_masks_list.append(low_res_masks)
        iou_predictions_list.append(iou_predictions[:, :1])
    low_res_masks = torch.cat(low_res_masks_list, dim=0)
    iou_predictions = torch.cat(iou_predictions_list, dim=0)
    mask_logits = F.interpolate(
        low_res_masks,
        size=(model.image_encoder.img_size, model.image_encoder.img_size),
        mode="bilinear",
        align_corners=False,
    )
    return mask_logits, iou_predictions


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numer = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((numer + eps) / (denom + eps)).mean()


def hard_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = logits > 0.0
    target_mask = targets > 0.5
    intersection = (preds & target_mask).sum(dim=(1, 2, 3)).float()
    union = (preds | target_mask).sum(dim=(1, 2, 3)).float()
    return (intersection + eps) / (union + eps)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    dice = 1.0 if dice_den == 0 else (2.0 * tp) / dice_den
    iou = 1.0 if iou_den == 0 else tp / iou_den
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "iou": float(iou),
    }


def evaluate_split(
    model: nn.Module,
    samples: Sequence[Sample],
    input_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.image_encoder.eval()
    model.prompt_encoder.eval()
    model.mask_decoder.eval()
    dice_values: List[float] = []
    iou_values: List[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    start = time.time()
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            gt_mask = np.asarray(Image.open(sample.mask_path).convert("L"), dtype=np.uint8) > 127
            resized_image = pil_resize_image(image, input_size)
            image_tensor = torch.from_numpy(np.asarray(resized_image, dtype=np.float32)).permute(2, 0, 1).contiguous()
            image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)
            mask_logits, _ = forward_no_prompt(model, image_tensor)
            upsampled_logits = F.interpolate(
                mask_logits,
                size=gt_mask.shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_mask = upsampled_logits[0, 0].cpu().numpy() > 0.0
            sample_metrics = binary_metrics(pred_mask, gt_mask)
            dice_values.append(sample_metrics["dice"])
            iou_values.append(sample_metrics["iou"])
            total_tp += int(sample_metrics["tp"])
            total_fp += int(sample_metrics["fp"])
            total_fn += int(sample_metrics["fn"])
    dice_dataset_den = 2 * total_tp + total_fp + total_fn
    iou_dataset_den = total_tp + total_fp + total_fn
    return {
        "num_samples": len(samples),
        "dice_mean": float(np.mean(dice_values)) if dice_values else 0.0,
        "iou_mean": float(np.mean(iou_values)) if iou_values else 0.0,
        "dice_dataset": 1.0 if dice_dataset_den == 0 else (2.0 * total_tp) / dice_dataset_den,
        "iou_dataset": 1.0 if iou_dataset_den == 0 else total_tp / iou_dataset_den,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "elapsed_sec": time.time() - start,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def append_history(path: Path, row: Dict[str, float], fieldnames: Sequence[str]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.model_type, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    args.checkpoint = str(checkpoint_path)
    output_dir = Path(args.output_dir).resolve()
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    summary_path = output_dir / "summary.json"
    args_path = output_dir / "args.json"
    args_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    dataset_root = Path(args.dataset_root).resolve()
    train_samples = load_split_samples(dataset_root, args.dataset_format, "train", args.train_limit)
    val_samples = load_split_samples(dataset_root, args.dataset_format, "val", args.val_limit)
    test_samples = load_split_samples(dataset_root, args.dataset_format, "test", args.test_limit)
    if not train_samples:
        raise RuntimeError("Training set is empty.")

    print(f"Dataset root: {dataset_root}")
    print(f"Train/Val/Test: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")
    print(f"Output dir: {output_dir}")
    print(f"Model type: sam_{args.model_type}")
    print(f"Input size: {args.input_size}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Selection metric: {SELECTION_METRIC}")
    print(f"Device: {device}")

    train_dataset = DCARingTrainDataset(train_samples, args.input_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = sam_model_registry[args.model_type](checkpoint=str(checkpoint_path)).to(device)
    set_sam_input_size(model, args.input_size)
    configure_trainable_params(model, args.train_mode)
    model.image_encoder.eval()
    model.prompt_encoder.eval()
    model.mask_decoder.train()
    if args.train_mode == "full":
        model.image_encoder.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    bce_loss = nn.BCEWithLogitsLoss()

    run_name = args.run_name or output_dir.name
    model_name = f"sam_{args.model_type}"
    current_dataset_tag = dataset_tag(args.dataset_format)
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        dir=str(wandb_dir),
        config={
            **vars(args),
            "model_name": model_name,
            "train_mode": args.train_mode,
            "freeze_image_encoder": args.train_mode == "decoder_only",
            "freeze_prompt_encoder": True,
            "train_mask_decoder": True,
            "prompt_mode": "no_prompt",
            "selection_metric": SELECTION_METRIC,
            "dataset_tag": current_dataset_tag,
        },
        tags=[
            model_name,
            current_dataset_tag,
            "no_prompt",
            args.train_mode,
            f"size_{args.input_size}",
        ],
    )
    wandb.save(str(args_path), policy="now")
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_bce_loss",
        "train_dice_loss",
        "train_iou_loss",
        "train_dice_dataset",
        "train_iou_dataset",
        "val_dice_mean",
        "val_iou_mean",
        "val_dice_dataset",
        "val_iou_dataset",
        "epoch_sec",
    ]
    wandb.save(str(history_path), policy="live")

    best_val_dice = -1.0
    best_epoch = -1

    try:
        for epoch in range(1, args.epochs + 1):
            model.image_encoder.eval()
            model.prompt_encoder.eval()
            model.mask_decoder.train()
            if args.train_mode == "full":
                model.image_encoder.train()
            epoch_start = time.time()
            total_loss_sum = 0.0
            total_bce_sum = 0.0
            total_dice_loss_sum = 0.0
            total_iou_loss_sum = 0.0
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_pixels = 0

            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                mask_logits, iou_predictions = forward_no_prompt(
                    model,
                    images,
                    freeze_image_encoder=args.train_mode == "decoder_only",
                )
                loss_bce = bce_loss(mask_logits, masks)
                loss_dice = dice_loss(mask_logits, masks)
                iou_targets = hard_iou_from_logits(mask_logits.detach(), masks)
                loss_iou = F.mse_loss(iou_predictions.squeeze(1), iou_targets)
                loss = loss_bce + loss_dice + 0.1 * loss_iou
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

                batch_size = images.shape[0]
                total_loss_sum += float(loss.item()) * batch_size
                total_bce_sum += float(loss_bce.item()) * batch_size
                total_dice_loss_sum += float(loss_dice.item()) * batch_size
                total_iou_loss_sum += float(loss_iou.item()) * batch_size
                preds = (mask_logits.detach() > 0.0)
                gts = (masks > 0.5)
                total_tp += int((preds & gts).sum().item())
                total_fp += int((preds & (~gts)).sum().item())
                total_fn += int(((~preds) & gts).sum().item())
                total_pixels += batch_size

            train_dice_den = 2 * total_tp + total_fp + total_fn
            train_iou_den = total_tp + total_fp + total_fn
            train_metrics = {
                "train_loss": total_loss_sum / max(total_pixels, 1),
                "train_bce_loss": total_bce_sum / max(total_pixels, 1),
                "train_dice_loss": total_dice_loss_sum / max(total_pixels, 1),
                "train_iou_loss": total_iou_loss_sum / max(total_pixels, 1),
                "train_dice_dataset": 1.0 if train_dice_den == 0 else (2.0 * total_tp) / train_dice_den,
                "train_iou_dataset": 1.0 if train_iou_den == 0 else total_tp / train_iou_den,
            }
            val_metrics = evaluate_split(model, val_samples, args.input_size, device)
            epoch_sec = time.time() - epoch_start
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_sec": epoch_sec,
                **train_metrics,
                "val_dice_mean": val_metrics["dice_mean"],
                "val_iou_mean": val_metrics["iou_mean"],
                "val_dice_dataset": val_metrics["dice_dataset"],
                "val_iou_dataset": val_metrics["iou_dataset"],
            }
            append_history(history_path, row, fieldnames)
            wandb.log(row, step=epoch)

            last_metrics = {
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_sec": epoch_sec,
            }
            save_checkpoint(
                ckpt_dir / "last.pt",
                model,
                optimizer,
                epoch,
                best_val_dice,
                last_metrics,
                args,
            )

            if val_metrics["dice_mean"] > best_val_dice:
                best_val_dice = float(val_metrics["dice_mean"])
                best_epoch = epoch
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val_dice,
                    last_metrics,
                    args,
                )

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} "
                f"loss={train_metrics['train_loss']:.6f} "
                f"train_dice={train_metrics['train_dice_dataset']:.4f} "
                f"val_dice_mean={val_metrics['dice_mean']:.4f} "
                f"val_iou_mean={val_metrics['iou_mean']:.4f} "
                f"time={epoch_sec:.1f}s"
            )

        best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_val_metrics = evaluate_split(model, val_samples, args.input_size, device)
        test_metrics = evaluate_split(model, test_samples, args.input_size, device)
        wandb.log(
            {
                "best_epoch": best_epoch,
                "best_val_dice_mean": best_val_metrics["dice_mean"],
                "best_val_iou_mean": best_val_metrics["iou_mean"],
                "test_dice_mean": test_metrics["dice_mean"],
                "test_iou_mean": test_metrics["iou_mean"],
                "test_dice_dataset": test_metrics["dice_dataset"],
                "test_iou_dataset": test_metrics["iou_dataset"],
            }
        )
        summary = {
            "model_type": model_name,
            "sam_model_key": args.model_type,
            "train_mode": args.train_mode,
            "freeze_image_encoder": args.train_mode == "decoder_only",
            "input_size": args.input_size,
            "selection_metric": SELECTION_METRIC,
            "best_epoch": best_epoch,
            "best_val_dice_mean": best_val_metrics["dice_mean"],
            "best_val_iou_mean": best_val_metrics["iou_mean"],
            "best_val_dice_dataset": best_val_metrics["dice_dataset"],
            "best_val_iou_dataset": best_val_metrics["iou_dataset"],
            "test_dice_mean": test_metrics["dice_mean"],
            "test_iou_mean": test_metrics["iou_mean"],
            "test_dice_dataset": test_metrics["dice_dataset"],
            "test_iou_dataset": test_metrics["iou_dataset"],
            "best_checkpoint": str((ckpt_dir / "best.pt").resolve()),
            "last_checkpoint": str((ckpt_dir / "last.pt").resolve()),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        wandb.save(str(summary_path), policy="now")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
