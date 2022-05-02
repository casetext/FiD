FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime


##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

##############################################################################
# Installation/Basic Utilities
##############################################################################


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc gnupg2 \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

##############################################################################
# Some Packages
##############################################################################
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen \
        libaio-dev

##############################################################################
# Installation/Python
##############################################################################

RUN apt-get install -y  python3.9 python3.9-dev python3-pip && \
    rm -f /usr/bin/python && \
    rm -rf /var/lib/apt/lists/* && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip && \
    # Print python an pip version
    python -V && pip -V
RUN pip install pyyaml
#RUN pip install ipython
RUN pip install typing_extensions


##############################################################################
# PyYAML build issue
# https://stackoverflow.com/a/53926898
##############################################################################
#RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
#        rm -rf /usr/lib/python3/dist-packages/PyYAML-*

##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
#RUN useradd --create-home --uid 8877 deepspeed
RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
#USER deepspeed



##############################################################################
# Installation Latest Git
##############################################################################
#RUN ln -s apt_pkg.cpython-{39m}-x86_64-linux-gnu.so
#RUN add-apt-repository ppa:git-core/ppa -y && \
#        apt-get update && \
#        apt-get install -y git && \
#        git --version

RUN apt-get update && \
    apt-get install -y git && \
    git --version




##############################################################################
# OpenMPI install
##############################################################################

ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.0
RUN mkdir -p ${STAGE_DIR}/build && \
    cd ${STAGE_DIR}/build && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ~ && \
    rm -rf ${STAGE_DIR}/build

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun


##############################################################################
# Rebuild pytorch
##############################################################################

ENV CMAKE_PREFIX_PATH="$(which cmake)"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV PYTORCH_BUILD_VERSION=1.11.0
ENV PYTORCH_BUILD_NUMBER=1
ENV USE_CUDA=1
ENV USE_CUDNN=1
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MAX_JOBS=4
RUN git clone --recursive https://github.com/pytorch/pytorch ${STAGE_DIR}/pytorch
RUN cd ${STAGE_DIR}/pytorch && \
    python setup.py clean && \
    python setup.py install

##############################################################################
# DeepSpeed
##############################################################################
RUN pip install triton==1.0.0
RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
RUN cd ${STAGE_DIR}/DeepSpeed && \
        git checkout . && \
        git checkout master
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_FUSED_LAMB=1 DS_BUILD_AIO=1 DS_BUILD_TRANSFORMER_INFERENCE=1 \
    pip install . --global-option="build_ext" --global-option="-j8" --no-cache -v \
    --disable-pip-version-check 2>&1  | tee build.log
#RUN rm -rf ${STAGE_DIR}/DeepSpeed
#RUN python -c "import deepspeed; print(deepspeed.__version__)"
RUN ds_report

RUN mkdir /app
COPY . /app/
WORKDIR /app
RUN pip install sentencepiece
RUN pip3 install -r requirements-docker.txt