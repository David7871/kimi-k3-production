# Usage: kdiff.py OFF_DIR ON_DIR : per-kernel-name ms/step on stage-0 TP-rank-0 (dev 0), names new or changed >0.05 ms
import sqlite3, glob, sys, collections
def prof(d):
    for f in sorted(glob.glob(d + "/rtx-01/*.sqlite")):
        c = sqlite3.connect(f)
        if not c.execute("select name from sqlite_master where name=\x27CUPTI_ACTIVITY_KIND_KERNEL\x27").fetchone(): continue
        if c.execute("select deviceId from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0] != 0: continue
        gl = [r[0] for r in c.execute("select r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like \x27cudaGraphLaunch%\x27 order by r.start")]
        a, b = gl[len(gl)//10], gl[-len(gl)//10]; n = sum(1 for t in gl if a <= t < b)
        acc = collections.Counter()
        for s, e, nm in c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where k.start>=? and k.end<?", (a, b)):
            acc[nm[:70]] += (e - s) / 1e6 / n
        return acc
A, B = prof(sys.argv[1]), prof(sys.argv[2])
for k in sorted(set(A) | set(B), key=lambda k: -abs(B[k] - A[k])):
    if abs(B[k] - A[k]) > 0.05: print("%-70s off %.3f on %.3f d %+.3f" % (k, A[k], B[k], B[k] - A[k]))
