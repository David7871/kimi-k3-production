#!/usr/bin/env python3
"""Delete LMCache entries written inside given time windows (and all salted
diagnostic entries) through the MP server's cache-control API.

The native filesystem L2 adapter names each object
``<model>@0x<kv_rank>@<group>@<chunk_hash>[@<cache_salt>].data`` with ``/``
replaced by ``-SEP-``; the file mtime is the L2 store time, which under the
write-through store controller is seconds after the prefill that produced
the entry. Entries whose mtime falls inside a window are deleted from both
tiers (``tier: all``) so that copies staged into L1 disappear with the L2
object and the adapter's capacity accounting stays consistent (deleting the
files directly would leave the eviction policy believing they exist).

Usage:
  lmcache_purge_window.py L2_DIR [--window 'YYYY-MM-DD HH:MM:SS' 'YYYY-MM-DD HH:MM:SS']...
                          [--salted] [--salt-prefix iso-]... [--execute] [--api http://127.0.0.1:8088]
Window bounds are UTC. Without --execute the script only reports counts.
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("l2_dir")
ap.add_argument("--window", nargs=2, action="append", default=[], metavar=("START", "END"))
ap.add_argument("--salted", action="store_true", help="also delete every salted (cache_salt != '') entry")
ap.add_argument("--salt-prefix", action="append", default=[], metavar="PREFIX",
                help="delete entries whose cache_salt starts with PREFIX (e.g. iso- for isolated namespaces); repeatable")
ap.add_argument("--execute", action="store_true")
ap.add_argument("--api", default="http://127.0.0.1:8088")
ap.add_argument("--batch", type=int, default=4000)
args = ap.parse_args()

windows = []
for s, e in args.window:
    ws = dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()
    we = dt.datetime.fromisoformat(e).replace(tzinfo=dt.timezone.utc).timestamp()
    windows.append((ws, we))


def parse_name(name):
    if not name.endswith(".data"):
        return None
    stem = name[:-5].replace("-SEP-", "/")
    parts = stem.split("@")
    if len(parts) < 4:
        return None
    model, rank_hex, group_hex, chunk_hex = parts[0], parts[1], parts[2], parts[3]
    salt = "@".join(parts[4:]) if len(parts) > 4 else ""
    return {"chunk_hash_hex": chunk_hex, "model_name": model, "kv_rank": int(rank_hex, 16),
            "object_group_id": int(group_hex, 16), "cache_salt": salt}


selected, total, sel_bytes, by_reason = [], 0, 0, {"window": 0, "salted": 0, "prefix": 0}
with os.scandir(args.l2_dir) as it:
    for entry in it:
        if not entry.is_file():
            continue
        total += 1
        st = entry.stat()
        key = parse_name(entry.name)
        if key is None:
            continue
        reason = None
        if args.salted and key["cache_salt"]:
            reason = "salted"
        elif any(key["cache_salt"].startswith(p) for p in args.salt_prefix):
            reason = "prefix"
        elif any(ws <= st.st_mtime <= we for ws, we in windows):
            reason = "window"
        if reason:
            selected.append(key)
            sel_bytes += st.st_size
            by_reason[reason] += 1

print(f"L2 objects: {total}; selected {len(selected)} ({sel_bytes/2**30:.1f} GiB) "
      f"window={by_reason['window']} salted={by_reason['salted']} prefix={by_reason['prefix']}", flush=True)
if not args.execute:
    print("dry run (pass --execute to delete)")
    sys.exit(0)

deleted = skipped = 0
for i in range(0, len(selected), args.batch):
    batch = selected[i:i + args.batch]
    body = json.dumps({"keys": batch, "tier": "all", "force": True}).encode()
    req = urllib.request.Request(args.api + "/cache/objects", data=body, method="DELETE",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    deleted += int(res.get("deleted", 0))
    skipped += int(res.get("skipped", 0))
    if not res.get("ok", True):
        print(f"batch {i//args.batch}: error {res.get('error')}", flush=True)
    print(f"batch {i//args.batch}: deleted={res.get('deleted')} skipped={res.get('skipped')} ok={res.get('ok')}", flush=True)
print(f"done: deleted={deleted} skipped={skipped}")
