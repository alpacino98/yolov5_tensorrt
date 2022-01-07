FROM nvcr.io/nvidia/tensorrt:19.10-py3

ARG DEBIAN_FRONTEND=noninteractive
#ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt install -y wget && apt-get install -y vim nano 
RUN apt-get install -y git pkg-config libopencv-dev
RUN apt install -y build-essential libssl-dev
RUN apt-get update

RUN mkdir /root/yolov5
COPY ./yolov5 /root/yolov5
WORKDIR /root/yolov5 