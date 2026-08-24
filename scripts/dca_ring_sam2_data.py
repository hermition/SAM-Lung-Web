"""DCA-ring adapter for the SAM2 single-frame training pipeline."""

from pathlib import Path

from training.dataset.vos_raw_dataset import VOSFrame, VOSRawDataset, VOSVideo
from training.dataset.vos_segment_loader import BinaryPNGSegmentLoader


class DCARingRawDataset(VOSRawDataset):
    """Expose each DCA-ring image as a one-frame, one-object video."""

    def __init__(self, dataset_root, split, sample_rate=1, truncate_video=-1, **kwargs):
        del kwargs
        root = Path(dataset_root).expanduser().resolve()
        image_dir = root / split / "imgs"
        mask_dir = root / split / "masks" / "0"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing DCA-ring split: {image_dir}, {mask_dir}")

        samples = []
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            samples.append((image_path, mask_path))
        samples = samples[:: max(int(sample_rate), 1)]
        if int(truncate_video) > 0:
            samples = samples[: int(truncate_video)]
        if not samples:
            raise RuntimeError(f"No DCA-ring samples found in {image_dir}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def get_video(self, idx):
        image_path, mask_path = self.samples[idx]
        video = VOSVideo(
            video_name=image_path.stem,
            video_id=idx,
            frames=[VOSFrame(frame_idx=0, image_path=str(image_path))],
        )
        segment_loader = BinaryPNGSegmentLoader({0: str(mask_path)})
        return video, segment_loader
