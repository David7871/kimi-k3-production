# lmcache_fs native extension with object integrity trailer

`lmcache_fs.cpython-312-x86_64-linux-gnu.so` is the LMCache native
filesystem connector built from fork branch
`agent/kimi-k3-fs-object-checksum-20260902` (LMCache PR #40, on the served
base a128b2e2): every object file ends with a 16-byte trailer (magic
`LMC1`, CRC-32C of the payload, payload length) verified on load; the
temporary file is fsynced before the hard-link publish and the directory
afterwards. Objects without a trailer (written before this build) load
unverified.

Mounted over
`/opt/venv/lib/python3.12/site-packages/lmcache/lmcache_fs.cpython-312-x86_64-linux-gnu.so`
by `candidates/k3-lmcache-fs-checksum-20260902/compose.lmcache-fs-checksum.override.yml`.

Rebuild (inside the production image, torch cpp_extension, same flags as
`setup_extensions/common_cpp.py`: `-O3 -std=c++17` + the torch C++ ABI flag):

```
docker run --rm -v <lmcache checkout>:/src:ro -v $PWD:/build --entrypoint sh -w /build \
  kimi-k3-production-issue75-dspark:k3-prefill-sm120-fa4-20260901-r3 \
  -c 'LMCACHE_SRC=/src /opt/venv/bin/python3 setup_fs.py build_ext --build-lib /build/out --build-temp /build/tmp'
```

Validation: `tests/v1/distributed/test_native_fs_object_checksum.py` and the
existing native FS tests (18 passed); `bench_fs.py` on the production L2
array (64 x 16 MiB, 4 workers): load 11.7 GiB/s, store 1.22 GiB/s with fsync.
