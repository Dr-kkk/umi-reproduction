# 步骤1: 选择一个包含Mambaforge的基础镜像
# Mambaforge是Conda的快速替代品，能更快地解决和安装依赖。
FROM condaforge/mambaforge:latest

# 步骤2: 设置工作目录
# 在容器内部创建一个 /app 目录，并将其设置为后续命令的执行目录。
WORKDIR /app

# 步骤3: 复制环境定义文件
# 将我们本地的 environment.yml 文件复制到容器的 /app 目录下。
# 这一步单独做可以利用Docker的缓存机制，如果环境文件不变，后续构建会更快。
COPY environment.yml .

# 步骤4: 创建并激活Conda环境
# 使用 environment.yml 文件在容器中创建名为 umi 的环境。
RUN mamba env create -f environment.yml

# 步骤5: 将Conda环境的激活脚本添加到shell配置中
# 这样，每次进入容器时，umi环境就会自动激活。
SHELL ["bash", "-c"]
RUN echo "conda activate umi" >> ~/.bashrc

# 步骤6: 复制你项目中的所有代码
# 将本地当前目录下的所有文件复制到容器的 /app 目录下。
COPY . .

# 步骤7: 设置默认命令
# 当容器启动时，默认执行 /bin/bash 命令，这样我们就可以进入一个交互式的终端。
CMD ["/bin/bash"]