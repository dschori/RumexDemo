# Dockerfile for RumexDemo U-Net training
# Based on the tensorflow docker image
# created by Damian Schori

FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

RUN pip install opencv-python && \
    pip install scikit-learn && \
    pip install pandas && \
    pip install seaborn

RUN mkdir -p /workspace

WORKDIR /workspace

RUN git clone https://github.com/dschori/realsense-ros.git