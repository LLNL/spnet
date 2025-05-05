# Base pytorch docker image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime 

# Apt packages
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    vim \
    gcc \
    wget \
    rsync \
    grsync \
    git

# Python packages
RUN pip install pytorch_metric_learning
RUN pip install torchvision
RUN pip install pycocotools
RUN pip install ray[tune]
RUN pip install albumentations==1.3.1
RUN pip install notebook
RUN pip install matplotlib
RUN pip install pandas
RUN pip install tensorflow
RUN pip install tensorboard
RUN pip install pytorch-lightning
RUN pip install lightning-bolts
RUN pip install torchmetrics
RUN pip install einops
RUN pip install higher
RUN pip install lightning-flash
RUN pip install jupyterlab
RUN pip install timm

# LoRA
RUN pip install git+https://github.com/microsoft/LoRA

# Install specific lightning package versions
RUN pip install lightning-bolts
RUN pip install lightning-flash==0.8.1.post0
RUN pip install lightning-utilities==0.8.0
RUN pip install pytorch-lightning==2.0.2

# Update torchvision
RUN pip install -U torchvision

# Install this package (OSR + SPNet)
## If you plan to re-install in the container, recommend commenting this out
COPY ./ /opt/spnet
WORKDIR /opt/spnet
RUN python3 setup.py install

# Environment variables
ENV TORCH_HOME=/weights
ENV CUDA_HOME=/opt/conda
