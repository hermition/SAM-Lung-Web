# 医院内网迁移清单

## 包含内容

- 网站与推理代码：`web_ui/`、`segment_anything/`
- 固定版本 SAM2 源码：Git submodule `external/sam2`
- 肺部分割推理权重：`web_ui/checkpoints/lung_sam2_hiera_l.pt`（Git LFS）
- pip 环境文件：`web_ui/requirements.txt`
- 启动脚本：`scripts/run_hospital_web.sh`
- 自动化测试：`tests/test_case_logger.py`

## 新服务器安装

```bash
git lfs install
git clone --recurse-submodules https://github.com/hermition/SAM-Lung-Web.git
cd SAM-Lung-Web
conda create -n sam-lung-web python=3.10 -y
conda activate sam-lung-web
pip install -r web_ui/requirements.txt
pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
bash scripts/run_hospital_web.sh
```

真实权重端到端验收可使用 `scripts/test_hospital_inference.py`，传入一对测试图片和 mask；该脚本会实际加载 SAM2、完成点 prompt 推理并核对 case 日志。

如服务器已 clone 但缺少权重或 SAM2 源码：

```bash
git lfs pull
git submodule update --init --recursive
```

默认使用 `cuda:0`、监听 `0.0.0.0:7860`，病例写入仓库下的 `hospital_cases/`。可在启动前覆盖：

```bash
SAM_DEVICE=cuda:1 \
SAM_WEB_OUTPUT_DIR=/data/sam_lung_cases \
GRADIO_SERVER_PORT=7860 \
bash scripts/run_hospital_web.sh
```

迁移验收至少确认：权重 SHA-256 一致；测试通过；上传后立即出现 `input.png` 和 `case.json`；每次点击新增一张 mask 和一条带时间的 prompt；“一键导出全部日志”能下载包含所有 case 的 ZIP；重启服务后旧 case 仍保留；医院反向代理已提供认证与 HTTPS。

```bash
cd web_ui/checkpoints
sha256sum -c SHA256SUMS
```
