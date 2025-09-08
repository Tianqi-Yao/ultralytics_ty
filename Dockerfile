# Dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

# 仅装必须：git + JupyterLab + 构建工具
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir -U pip setuptools wheel jupyterlab

ENV PYTHONPATH=/workspace
CMD ["bash"]
