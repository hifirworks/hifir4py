#!/bin/bash

# Install openblas
yum install openblas-devel -y
export HIFIR_LAPACK_LIB=-lopenblasp

# Compile wheels
pys=(cp36-cp36m cp37-cp37m cp38-cp38 cp39-cp39)
for PY in ${pys[@]}; do
    "/opt/python/${PY}/bin/pip" install -r /hifir4py/requirements.txt
    "/opt/python/${PY}/bin/pip" wheel /hifir4py/ --no-deps -w /hifir4py/temp-wheelhouse/
done

# Repair binary wheels
for whl in /hifir4py/temp-wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat "$PLAT" -w /hifir4py/dist/
done

# Chown
chown -R $HOST_UID:$HOST_GID /hifir4py/dist/

rm -rf /hifir4py/temp-wheelhouse
