ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${FROM_IMAGE_NAME}

COPY . /hifi_vc
WORKDIR /hifi_vc
ENV PYTHONPATH /hifi_vc

ADD requirements.txt .
RUN pip install -r requirements.txt
