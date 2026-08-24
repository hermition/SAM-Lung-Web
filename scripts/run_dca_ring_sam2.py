#!/usr/bin/env python
"""Launch one prompt-free SAM2 DCA-ring experiment through the SAM2 trainer."""

import argparse
import os
import random
import sys
from pathlib import Path

from hydra import compose, initialize_config_module
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = REPO_ROOT / "external" / "sam2"
BASE_CONFIG = "configs/sam2_training/lung_hiera_l_orig_decoder_noprompt_freezeenc_50ep"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-mode", choices=("decoder_only", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", default="sam2-dca-ring")
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def replace_raw_dataset(node: DictConfig, dataset_root: str, split: str) -> None:
    if not isinstance(node, DictConfig):
        return
    target = str(node.get("_target_", ""))
    if target.endswith("LungManifestRawDataset"):
        with open_dict(node):
            node["_target_"] = "scripts.dca_ring_sam2_data.DCARingRawDataset"
            node["dataset_root"] = dataset_root
            node["split"] = split
            node.pop("manifest_csv", None)
        return
    for value in node.values():
        if isinstance(value, DictConfig):
            replace_raw_dataset(value, dataset_root, split)
        elif isinstance(value, ListConfig):
            for item in value:
                if isinstance(item, DictConfig):
                    replace_raw_dataset(item, dataset_root, split)


def configure(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["SAM2_ORIG_DECODER_RUN_NAME"] = args.run_name
    os.environ["SAM2_ORIG_DECODER_RUN_DIR"] = str(Path(args.run_dir).resolve())
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    with initialize_config_module("sam2", version_base="1.2"):
        cfg = compose(config_name=BASE_CONFIG)

    run_dir = Path(args.run_dir).resolve()
    cfg.scratch.resolution = args.input_size
    cfg.scratch.train_batch_size = args.batch_size
    cfg.scratch.val_batch_size = args.batch_size
    cfg.scratch.num_epochs = args.epochs
    cfg.scratch.base_lr = args.lr
    cfg.scratch.vision_lr = args.lr
    cfg.scratch.num_train_workers = args.num_workers
    cfg.scratch.num_val_workers = args.num_workers
    cfg.scratch.num_frames = 1
    cfg.scratch.max_num_objects = 1
    cfg.dataset.multiplier = 1

    for phase in ("train", "val", "test"):
        replace_raw_dataset(cfg.trainer.data[phase], args.dataset_root, phase)

    model_cfg = cfg.trainer.model
    model_cfg.image_size = args.input_size
    model_cfg.disable_prompt_inputs = True
    model_cfg.prob_to_use_pt_input_for_train = 0.0
    model_cfg.prob_to_use_pt_input_for_eval = 0.0
    model_cfg.prob_to_use_box_input_for_train = 0.0
    model_cfg.prob_to_use_box_input_for_eval = 0.0
    model_cfg.prob_to_sample_from_gt_for_train = 0.0
    model_cfg.num_correction_pt_per_frame = 0
    model_cfg.num_frames_to_correct_for_train = 1
    model_cfg.num_frames_to_correct_for_eval = 1
    model_cfg.num_init_cond_frames_for_train = 1
    model_cfg.num_init_cond_frames_for_eval = 1
    model_cfg.forward_backbone_per_frame_for_eval = True
    model_cfg.use_mask_input_as_output_without_sam = False
    model_cfg.freeze_image_encoder = args.train_mode == "decoder_only"
    model_cfg.train_only_sam_mask_decoder = args.train_mode == "decoder_only"
    cfg.trainer.distributed.find_unused_parameters = args.train_mode == "full"

    cfg.trainer.seed_value = args.seed
    cfg.trainer.mode = "train"
    cfg.trainer.logging.log_dir = str(run_dir / "logs")
    cfg.trainer.logging.tensorboard_writer.log_dir = str(run_dir / "tensorboard")
    wandb_cfg = cfg.trainer.logging.wandb_writer
    wandb_cfg.project = args.wandb_project
    wandb_cfg.name = args.run_name
    wandb_cfg.dir = str(run_dir / "wandb")
    wandb_cfg.mode = args.wandb_mode
    with open_dict(wandb_cfg.config):
        wandb_cfg.config.model = "sam2_hiera_large_native_decoder"
        wandb_cfg.config.train_mode = args.train_mode
        wandb_cfg.config.prompt_free = True
        wandb_cfg.config.epochs = args.epochs
        wandb_cfg.config.resolution = args.input_size
        wandb_cfg.config.batch_size = args.batch_size
        wandb_cfg.config.base_lr = args.lr
        wandb_cfg.config.vision_lr = args.lr
        wandb_cfg.config.freeze_image_encoder = args.train_mode == "decoder_only"
        wandb_cfg.config.train_only_sam_mask_decoder = args.train_mode == "decoder_only"

    cfg.trainer.checkpoint.save_dir = str(run_dir / "checkpoints")
    cfg.launcher.experiment_log_dir = str(run_dir)
    cfg.launcher.gpus_per_node = 1
    cfg.launcher.num_nodes = 1
    cfg.submitit.use_cluster = False
    return cfg


def main():
    args = parse_args()
    if not SAM2_ROOT.is_dir():
        raise FileNotFoundError(f"Missing SAM2 source: {SAM2_ROOT}")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = configure(args)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    from training.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    OmegaConf.resolve(cfg)
    (run_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg), encoding="utf-8"
    )

    from training.train import single_proc_run
    print(f"Run: {args.run_name}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Mode: {args.train_mode}")
    print(f"Epochs/LR/BS/size: {args.epochs}/{args.lr}/{args.batch_size}/{args.input_size}")
    print(f"GPU (visible index): {args.gpu}")
    single_proc_run(
        local_rank=0,
        main_port=random.randint(10000, 65000),
        cfg=cfg,
        world_size=1,
    )


if __name__ == "__main__":
    main()
