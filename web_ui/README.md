# Custom SAM1 交互式网页分割

这是一个基于当前仓库 `SamPredictor` 的最小 Gradio 网页界面，支持：

- 上传 JPG、JPEG、PNG 等 2D 原图；
- 前景点/背景点点击 prompt；
- 多点、撤销、清空和重置；
- mask overlay、二值 mask 和 PNG 下载；
- 标准 SAM1 原始 checkpoint，以及训练生成的 `best.pt`/`last.pt`（包含 `model_state_dict`）。

## 启动

建议使用已有的 GPU 环境 `segllm`，无需修改或覆盖已有实验结果：

```bash
conda activate /data_new/moyancheng/envs/segllm
pip install -r web_ui/requirements.txt
python web_ui/app.py \
  --checkpoint /path/to/custom/best.pt \
  --model-type vit_b \
  --device cuda:0 \
  --input-size 1024
```

如果 checkpoint 的 `image_encoder.pos_embed` 能够推断输入边长，可以省略 `--input-size`；对于训练时使用 256 输入的模型，请显式传入 `--input-size 256`。浏览器访问 `http://localhost:7860`。

也可以使用环境变量配置：

```bash
SAM_CHECKPOINT=/path/to/custom/best.pt \
SAM_MODEL_TYPE=vit_b \
SAM_DEVICE=cuda:0 \
SAM_INPUT_SIZE=1024 \
python web_ui/app.py
```

输出默认保存到 `web_outputs/<session>/`，不会写入 `tmp/` 下的训练实验目录。

`--share` 仅适合临时演示；处理医疗或其他敏感图片时请使用内网地址，并在生产环境配置访问控制和反向代理。

## 交互说明

1. 上传原图，等待“图片已加载”；
2. 选择“前景点”，点击需要保留的目标；
3. 如需排除区域，切换为“背景点”后点击该区域；
4. 使用“撤销最后一点”或“清空 prompt”修正结果；
5. 点击“保存并下载结果”获取二值 mask 和 overlay PNG。

网页后端会为每个浏览器 session 单独缓存 image embedding，同一张图片的后续点击只运行 mask decoder。
