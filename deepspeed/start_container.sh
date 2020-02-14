#!/bin/bash

name=${1-deepspeed}
image=deepspeed/deepspeed:latest
echo "starting docker image named $name"
docker run -d -t --name $name \
        --network host \
        -v "/home/jspijkervet/git/Greedy_InfoMax":/home/deepspeed/Greedy_InfoMax \
        -v ${HOME}/.ssh:/home/deepspeed/.ssh \
        -v /job/hostfile:/job/hostfile \
        --gpus all $image bash -c 'sudo service ssh start && sleep infinity'
