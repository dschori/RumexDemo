# Dockerfile for RumexDemo U-Net training
# created by Damian Schori

FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# install openc-cv
RUN apt update && \
    apt install -y python3-opencv

# install python packages
RUN pip install scikit-learn && \
    pip install pandas && \
    pip install seaborn && \
    pip install albumentations

# clone git project
RUN mkdir -p /workspace
WORKDIR /workspace
RUN git clone https://github.com/dschori/RumexDemo.git

WORKDIR /workspace/RumexDemo