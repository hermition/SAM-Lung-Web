# SAM1/SAM2 交互式网页分割

这是一个基于 Gradio 的 2D 图像点击式分割网页界面。默认使用肺部数据集上进行 prompt-aware 微调的 SAM2 Hiera-L checkpoint，支持：

- 上传 JPG、JPEG、PNG 等 2D 原图；
- 前景点/背景点点击 prompt；
- 多点交互，后续点击复用上一轮 low-resolution mask；
- 撤销、清空和重置；
- mask overlay、二值 mask 和 PNG 下载；
- 上传时自动创建带 UTC 时间戳的 case，每次打点立即保存坐标、时间和对应 mask；
- 默认 SAM2 checkpoint，以及显式选择后端的 SAM1 checkpoint。

## 启动

建议在医院服务器创建独立环境，避免修改或覆盖已有实验结果：

```bash
conda create -n sam-lung-web python=3.10 -y
conda activate sam-lung-web
pip install -r web_ui/requirements.txt
pip install -e .
git submodule update --init --recursive
bash scripts/run_hospital_web.sh
```

默认 checkpoint 为：

```text
web_ui/checkpoints/lung_sam2_hiera_l.pt
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

部署脚本默认保存到 `hospital_cases/<UTC时间戳>_<随机后缀>/`，不会写入或覆盖训练实验目录。每个目录是一份独立 case：

```text
hospital_cases/20260909T120000.123456Z_a1b2c3d4/
├── case.json
├── input.png
├── masks/
│   ├── 0001_20260909T120010.000000Z.png
│   └── 0002_20260909T120015.000000Z.png
├── final_mask.png
└── final_overlay.png
```

`case.json` 是追加式审计日志，包含输入图像摘要、模型信息、每次点击的像素坐标/前背景标签/UTC 时间、当时生效的全部 prompt、score 和对应 mask 路径。撤销和清空也会记录，历史点击不会被删除。即使医生没有点击“保存结果”，输入和每次点击的 mask 也已经落盘。

`--share` 仅适合临时演示；处理医疗或其他敏感图片时请使用内网地址，并在生产环境配置访问控制和反向代理。

## 交互说明

1. 上传原图，等待“图片已加载”；
2. 选择“前景点”，点击需要保留的目标；
3. 如需排除区域，切换为“背景点”后点击该区域；
4. 使用“撤销最后一点”或“清空 prompt”修正结果；
5. 点击“保存结果”额外生成 `final_mask.png` 和 `final_overlay.png`。

网页后端会为每个浏览器 session 单独缓存 image embedding，同一张图片的后续点击只运行 prompt encoder 和 mask decoder。浏览器 session 只以 SHA-256 摘要写入日志，图片统一转存为 PNG，不保留上传文件的 EXIF 元数据。case 目录权限为 `0700`、文件权限为 `0600`。

医院部署时应将 `SAM_WEB_OUTPUT_DIR` 指向受控的数据盘，并由医院侧配置身份认证、HTTPS、访问审计、备份、留存/删除策略和磁盘加密。不要启用 `--share`，也不要把 `hospital_cases` 提交到 Git。SAM2 Hiera-L 约需要较大 GPU 显存；若启动时报依赖错误，请确认已执行 `git submodule update --init --recursive`。
