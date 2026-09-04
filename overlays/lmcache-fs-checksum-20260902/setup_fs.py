"""Build only the lmcache_fs extension from a source checkout (torch cpp_extension)."""
import os, sys
from setuptools import setup
from torch.utils import cpp_extension
src = os.environ["LMCACHE_SRC"]
abi = "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if __import__("torch")._C._GLIBCXX_USE_CXX11_ABI else "0")
ext = cpp_extension.CppExtension(
    "lmcache.lmcache_fs",
    sources=[f"{src}/csrc/storage_backends/fs/pybind.cpp", f"{src}/csrc/storage_backends/fs/connector.cpp"],
    include_dirs=[f"{src}/csrc/storage_backends", f"{src}/csrc/storage_backends/fs"],
    extra_compile_args={"cxx": [abi, "-O3", "-std=c++17"]},
)
setup(name="lmcache_fs_only", ext_modules=[ext], cmdclass={"build_ext": cpp_extension.BuildExtension})
