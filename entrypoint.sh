#!/bin/bash
set -e

# 设置 Conda 环境的路径
# 注意：mambaforge 镜像默认会将环境安装在 /opt/conda/envs/ 目录下
CONDA_ENV_PATH="/opt/conda/envs/umi"

# 检查环境是否存在，如果不存在，则不执行激活
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Activating Conda environment: umi"
    source activate umi
else
    echo "Warning: Conda environment 'umi' not found. Skipping activation."
fi

exec "$@"