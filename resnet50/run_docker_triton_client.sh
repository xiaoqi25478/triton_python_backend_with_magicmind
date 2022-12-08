#/bin/bash
set -e
set -x

export MY_CONTAINER="XIAOQI_TRITON_RESNET50_CLIENT_2206_PY3"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
if [ 0 -eq $num ];then
docker run  -e DISPLAY=unix$DISPLAY  \
	  --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 \
    --net=host --pid=host -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix \
    -dit --privileged --name $MY_CONTAINER --ipc=host \
    -v /tools/:/tools/ \
    -v /mnt/:/mnt/ \
    -v /data/:/data/ \
  	-v /usr/bin/cnmon:/usr/bin/cnmon \
	  nvcr.io/nvidia/tritonserver:22.06-py3-sdk /bin/bash
    docker start $MY_CONTAINER
    docker exec -ti $MY_CONTAINER /bin/bash
else
  docker start $MY_CONTAINER
  docker exec -ti $MY_CONTAINER /bin/bash
fi
