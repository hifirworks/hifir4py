#!/bin/sh

# Install openblas
yum install openblas-devel -y
export HIFIR_LAPACK_LIB=-lopenblasp

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /hifir4py/requirements.txt
    "${PYBIN}/pip" wheel /hifir4py/ --no-deps -w /hifir4py/temp-wheelhouse/
done

# Repair binary wheels
for whl in /hifir4py/temp-wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat "$PLAT" -w /hifir4py/wheelhouse/
done

rm -rf /hifir4py/temp-wheelhouse
