
ARG PYTORCH="2.3.1"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

#setup environment
ENV FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    PATH="/usr/local/cuda/bin:$PATH" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" MMCV_WITH_OPS="1"

# installation
RUN apt update \
        && apt install -y git libgl1 libglib2.0-0 ffmpeg libturbojpeg libsm6 libxext6 ninja-build libxrender-dev \
        && apt clean \
        && rm -rf /var/lib/apt/lists/* \
        && git clone https://github.com/open-mmlab/mmcv.git -b v2.1.0 mmcv-v2.1.0 \
        && cd mmcv-v2.1.0 \
        && pip install --no-cache-dir -r ./requirements/build.txt \
        && pip install --no-cache-dir .[all] -v \
        && cd .. \
        && git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 mmdet-v3.3.0 \
        && cd mmdet-v3.3.0 \
        && pip install --no-cache-dir -r ./requirements/build.txt \
        && pip install --no-cache-dir . -v \
        && cd .. \
        && rm -rf ./mmcv-v2.1.0  ./mmdet-v3.3.0


RUN groupadd -g ${GID} ${GROUPNAME} -f \
    && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} \
    && echo 'export PATH=~/.local/bin:${PATH}' >> /home/${USERNAME}/.bashrc
USER ${USERNAME}
RUN pip install -U cvtk
WORKDIR /home/${USERNAME}
