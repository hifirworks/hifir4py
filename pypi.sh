#!/bin/sh
set -e

# Build sdist
python3 setup.py sdist

export PLAT=manylinux2010_x86_64
export IMAGE=quay.io/pypa/$PLAT

# Pull the image
docker pull $IMAGE

# Build
docker run --rm -it -e PLAT=$PLAT -e HOST_UID=`id -u $USER` -e HOST_GID=`id -g $USER` -v `pwd`:/hifir4py $IMAGE /hifir4py/wheels/build.sh
