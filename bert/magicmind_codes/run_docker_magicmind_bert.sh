#/bin/bash
#--device=/dev/video0
# apt-get update && apt-get install gdb htop -y
export MY_CONTAINER="XIAOQI_TRITON_BERT_MAGICMIND0.13-UBUNTU18.04"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
if [ 0 -eq $num ];then
docker run  -e DISPLAY=unix$DISPLAY  \
    --net=host --pid=host -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix \
    -dit --privileged --name $MY_CONTAINER --ipc=host \
    -v /tools/:/tools/ \
    -v /mnt/:/mnt/ \
    -v /data/:/data/ \
    -v /usr/bin/comon:/usr/bin/comon \
    yellow.hub.cambricon.com/magicmind/release/x86_64/magicmind:0.13.0-x86_64-ubuntu18.04-py_3_7 /bin/bash
    docker start $MY_CONTAINER
    docker exec -ti $MY_CONTAINER /bin/bash
else
  docker start $MY_CONTAINER
  docker exec -ti $MY_CONTAINER /bin/bash
fi

  
