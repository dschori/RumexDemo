# Dockerfile for RumexDemo U-Net training

FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

# install openc-cv
RUN apt update && \
    apt install -y python3-opencv %% \
    apt install -y git && \
    apt install -y wget && \
    apt install -y unzip

# install python packages
RUN pip install scikit-learn && \
    pip install pandas && \
    pip install seaborn && \
    pip install albumentations && \
    pip install jupyterlab && \
    pip install segmentation_models_pytorch && \
    pip install ipywidgets
