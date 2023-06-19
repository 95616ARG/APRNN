#! /usr/bin/env bash

docker run --rm -it \
    "$@" \
    -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
    -v `pwd`:/host_aprnn_pldi23:rw \
    -v `pwd`/data:/aprnn_pldi23/data:rw \
    -v `pwd`/results:/aprnn_pldi23/results:rw \
    -w /aprnn_pldi23 \
    --privileged \
    aprnn_pldi23:dev
