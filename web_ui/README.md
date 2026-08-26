# SAM1/SAM2 交互式网页分割

这是一个基于 Gradio 的 2D 图像点击式分割网页界面。默认使用肺部数据集上进行 prompt-aware 微调的 SAM2 Hiera-L checkpoint，支持：

- 上传 JPG、JPEG、PNG 等 2D 原图；
- 前景点/背景点点击 prompt；
- 多点交互，后续点击复用上一轮 low-resolution mask；
- 撤销、清空和重置；
- mask overlay、二值 mask 和 PNG 下载；
- 默认 SAM2 checkpoint，以及显式选择后端的 SAM1 checkpoint。

## 启动

建议使用已有的 GPU 环境 `segllm`，无需修改或覆盖已有实验结果：

```bash
conda activate /data_new/moyancheng/envs/segllm
pip install -r web_ui/requirements.txt
python web_ui/app.py
```

默认 checkpoint 为：

```text
external/sam2/sam2_logs/lung_hiera_l_1024_f1_bs7_12ep_full_valbest_v2/
checkpoints/val_all_seg_slice_iou_mean.pt
```

默认 SAM2 配置为 `external/sam2/sam2/configs/sam2/sam2_hiera_l.yaml`，浏览器访问 `http://localhost:7860`。

如果要使用 SAM1 checkpoint，请显式指定后端：

```bash
python web_ui/app.py \
  --backend sam1 \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device cuda:0
```

也可以使用环境变量配置：

```bash
SAM_BACKEND=auto \
SAM_CHECKPOINT=/path/to/checkpoint.pt \
SAM_DEVICE=cuda:0 \
SAM2_CONFIG=configs/sam2/sam2_hiera_l.yaml \
python web_ui/app.py
```

`--model-type` 和 `--input-size` 主要用于 SAM1；SAM2 的输入尺寸由配置文件确定（当前默认是 1024）。`auto` 会识别仓库内默认 SAM2 路径；对于其他 `.pt` 文件，可以显式传入 `--backend sam2` 或 `--backend sam1`。

输出默认保存到 `web_outputs/<session>/`，不会写入 `tmp/` 下的训练实验目录。

`--share` 仅适合临时演示；处理医疗或其他敏感图片时请使用内网地址，并在生产环境配置访问控制和反向代理。

## 交互说明

1. 上传原图，等待“图片已加载”；
2. 选择“前景点”，点击需要保留的目标；
3. 如需排除区域，切换为“背景点”后点击该区域；
4. 使用“撤销最后一点”或“清空 prompt”修正结果；
5. 点击“保存并下载结果”获取二值 mask 和 overlay PNG。

网页后端会为每个浏览器 session 单独缓存 image embedding，同一张图片的后续点击只运行 prompt encoder 和 mask decoder。SAM2 Hiera-L 约需要较大 GPU 显存，若启动时报依赖错误，请确认 `external/sam2` 源码存在且环境安装了 `hydra-core`、`iopath`。
