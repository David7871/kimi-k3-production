"""Store/load throughput of the native FS connector: N objects of SIZE bytes."""
import os, select, sys, time
from lmcache import lmcache_fs
root, n, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
client = lmcache_fs.LMCacheFSClient(root, 4)
def wait():
    p = select.poll(); p.register(client.event_fd(), select.POLLIN)
    while True:
        if p.poll(50):
            c = client.drain_completions()
            if c: return c
bufs = [bytearray(os.urandom(size)) for _ in range(n)]
keys = [f"model@00000000@0@{i:064x}" for i in range(n)]
t0 = time.time(); client.submit_batch_set(keys, [memoryview(b) for b in bufs]); c = wait(); dt = time.time() - t0
print(f"store {n}x{size>>20}MiB: {dt:.2f}s = {n*size/dt/2**30:.2f} GiB/s ok={c[0][1]}")
os.system(f"sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true")
outs = [bytearray(size) for _ in range(n)]
t0 = time.time(); client.submit_batch_get(keys, [memoryview(o) for o in outs]); c = wait(); dt = time.time() - t0
print(f"load  {n}x{size>>20}MiB: {dt:.2f}s = {n*size/dt/2**30:.2f} GiB/s ok={c[0][1]} allok={all(c[0][3])} match={all(bytes(o)==bytes(b) for o,b in zip(outs,bufs))}")
client.close()
