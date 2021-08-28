# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HIFIR4PY project                       #
###############################################################################

import os
import glob
import tempfile
import zipfile
import shutil
import atexit
import urllib.request
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


_version = "0.1.0"


def get_hifir():
    tmp_path = tempfile.gettempdir()
    with urllib.request.urlopen(
        "https://github.com/hifirworks/hifir/archive/refs/tags/v{}.zip".format(_version)
    ) as url_f:
        with open(os.path.join(tmp_path, "hifir.zip"), "wb") as out_f:
            out_f.write(url_f.read())
    zipfile.ZipFile(os.path.join(tmp_path, "hifir.zip")).extractall(tempfile.gettempdir())
    # remove zip
    shutil.rmtree(os.path.join(tmp_path, "hifir.zip"), ignore_errors=True)
    hifir_include = os.path.join(tmp_path, "hifir-{}".format(_version), "src")

    def rm_hifir_src():
        shutil.rmtree(os.path.join(tmp_path, "hifir-{}".format(_version)), ignore_errors=True)

    atexit.register(rm_hifir_src)
    return hifir_include


def is_debug():
    flag = os.environ.get("HIFIR4PY_DEBUG", None)
    if flag is None:
        return False
    return flag.lower() not in ("0", "no", "off", "false")


_hifir4py_debug = is_debug()

incs = [".", get_hifir()]

# configure libraries
_lapack_lib = os.environ.get("HIFIR_LAPACK_LIB", "-llapack")
_lapack_libs = _lapack_lib.split(" ")
_lapack_a = os.environ.get("HIFIR_LAPACK_STATIC_LIB", None)
extra_objs = []
libs = []
if _lapack_a is not None and _lapack_a:
    _lapack_libs = []
    extra_objs = _lapack_a.split(" ")
    # For static linking we need to link to fortran runtime
    libs = ["gfortran"]
for i, _l in enumerate(_lapack_libs):
    if _lapack_libs[i].startswith("-l"):
        _lapack_libs[i] = _l[2:]
libs += _lapack_libs

# configure library paths
lib_dirs = None
_lapack_path = os.environ.get("HIFIR_LAPACK_LIB_PATH", "")
if _lapack_path:
    lib_dirs = [_lapack_path]
rpath = None if lib_dirs is None else lib_dirs


class BuildExt(build_ext):
    def _remove_flag(self, flag):
        try:
            self.compiler.compiler_so.remove(flag)
        except (AttributeError, ValueError):
            pass

    def build_extensions(self):
        self._remove_flag("-Wstrict-prototypes")
        opts = []
        if _hifir4py_debug:
            self._remove_flag("-DNDEBUG")
            opts.append("-DHIF_DEBUG")

        cpl_type = self.compiler.compiler_type

        def test_switch(flag):
            with tempfile.NamedTemporaryFile("w", suffix=".cpp") as tmp_f:
                tmp_f.write("int main(int argc, char *argv[]){return 0;}")
                try:
                    self.compiler.compile([tmp_f.name], extra_postargs=[flag])
                except BaseException:
                    return False
            return True

        if cpl_type == "unix":
            assert test_switch("-std=c++11"), "must have C++11 support"
            if test_switch("-std=c++1z"):
                opts.append("-std=c++1z")
            else:
                opts.append("-std=c++11")
            if test_switch("-ffast-math"):
                opts.append("-ffast-math")
            if test_switch("-rdynamic"):
                opts.append("-rdynamic")
            if test_switch("-O3") and "-O3" not in self.compiler.compiler_so:
                opts.append("-O3")
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()


_pyx = glob.glob(os.path.join("hifir4py", "*.pyx"))
_pyx += glob.glob(os.path.join("hifir4py", "_hifir", "*.pyx"))
_pyx += glob.glob(os.path.join("hifir4py", "ksp", "*.pyx"))
exts = []

for f in _pyx:
    _f = f.split(".")[0]
    mod = ".".join(_f.split(os.sep))
    exts.append(
        Extension(
            mod,
            [f],
            language="c++",
            include_dirs=incs,
            libraries=libs,
            library_dirs=lib_dirs,
            runtime_library_dirs=rpath,
            extra_objects=extra_objs,
        )
    )

_opts = {"language_level": 3, "embedsignature": True}
if not _hifir4py_debug:
    _opts.update({"wraparound": False, "boundscheck": False})
exts = cythonize(exts, compiler_directives=_opts)
