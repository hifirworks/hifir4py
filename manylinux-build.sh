#!/bin/sh
set -e

export PLAT=manylinux_2_12_x86_64
# NOTE the above is alias of the following by PEP 600
export IMAGE=quay.io/pypa/manylinux2010_x86_64

# Pull the image
docker pull $IMAGE

# Build
docker run --rm -it -e PLAT=$PLAT -v `pwd`:/hifir4py $IMAGE /hifir4py/wheels/build.sh
