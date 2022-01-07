!bin/bash

docker build . -t alphapro98/yolov5-tensorrt-alpha:0.1

#XSOCK=/tmp/.X11-unix
#XAUTH=$HOME/.Xauthority

#xhost +local:docker

VOLUMES="--volume=/mnt:/mnt:rw"

docker run -it --ipc=host $VOLUMES --privileged --runtime=nvidia  alphapro98/yolov5-tensorrt-alpha:0.1
