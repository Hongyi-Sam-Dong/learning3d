# CMU 16-825 Learning for 3D — GPU environment for NVIDIA DGX Spark (GB10, aarch64)
#
# Tested stack:
#   base        nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04 (arm64)
#   python      3.10 (Ubuntu 22.04 system python, isolated in /opt/venv)
#   torch       2.10.0+cu130, torchvision 0.25.0+cu130
#   pytorch3d   0.7.9, built from source for sm_121 (GB10)
#
# The repository is NOT copied into the image; compose.yaml bind-mounts it at
# /workspace/learning3d.

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG PYTORCH3D_REF=v0.7.9
# GB10 is compute capability 12.1
ARG TORCH_CUDA_ARCH_LIST=12.1
ARG MAX_JOBS=16
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    MPLBACKEND=Agg

# System packages: python toolchain, compilers for the PyTorch3D CUDA extension,
# git/git-lfs/unzip for fetching PyTorch3D and the R2N2 dataset.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        build-essential ninja-build \
        git git-lfs unzip ca-certificates \
        less nano \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --upgrade pip setuptools wheel

# GPU-enabled PyTorch / torchvision (pinned; later installs are constrained to these)
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL} \
    && printf "torch==%s\ntorchvision==%s\n" "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /opt/torch-constraints.txt

# PyTorch3D from source with CUDA kernels. No GPU is visible during `docker build`,
# so FORCE_CUDA is required. CUDA 13's nvcc defaults to
# -static-global-template-stub=true, which breaks linking of PyTorch3D's templated
# pulsar kernels (undefined pulsar::Renderer::* symbols); turn it off.
RUN FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_FLAGS="-static-global-template-stub=false" \
    pip install --no-build-isolation -c /opt/torch-constraints.txt \
        "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

# Assignment 2 requirements (torch, torchvision, numpy<2, PyMCubes, matplotlib)
# plus modules imported by the assignment code (imageio, scipy, tabulate, tqdm).
# Constrained so the CUDA torch/torchvision wheels cannot be replaced.
RUN pip install -c /opt/torch-constraints.txt \
        "numpy<2.0.0" PyMCubes matplotlib \
        imageio scipy tabulate tqdm \
    && python -c "import torch, torchvision, pytorch3d, mcubes; \
assert torch.version.cuda, 'CPU-only torch'; \
from pytorch3d import _C; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, '| pytorch3d', pytorch3d.__version__)"

# Non-root user matching the host user so files written to the bind mount keep
# host ownership. Owning the venv lets the user `pip install` extras if needed.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && mkdir -p /workspace/learning3d /home/${USERNAME}/.cache \
    && chown -R ${USER_UID}:${USER_GID} /workspace /home/${USERNAME} $VIRTUAL_ENV \
    && printf '%s\n' \
        'export PS1="\[\e[1;32m\](learning3d)\[\e[0m\] \u@\h:\[\e[1;34m\]\w\[\e[0m\]\$ "' \
        'alias ll="ls -alF"' \
        'alias gpucheck="python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))\""' \
        >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /workspace/learning3d

CMD ["bash"]
