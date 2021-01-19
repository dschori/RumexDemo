# Dockerfile for RumexDemo U-Net training
# Based on the tensorflow docker image
# created by Damian Schori

FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive

# install openc-cv
RUN apt update && \
    apt install -y python3-opencv

# install python packages
RUN pip install scikit-learn && \
    pip install pandas && \
    pip install seaborn

# clone git project
RUN mkdir -p /workspace
WORKDIR /workspace
RUN git clone https://github.com/dschori/RumexDemo.git

WORKDIR /workspace/RumexDemo