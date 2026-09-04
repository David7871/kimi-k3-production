#!/usr/bin/env python3
"""Decide the LMCache cache namespace of a Kimi-K3 vLLM boot from its configuration.

Every LMCache key written by a server carries the namespace as the leading
component of its cache_salt (LMCache connector env ``LMCACHE_CACHE_NAMESPACE``).
Entries written under one namespace are unreachable from another, so a boot
whose numerics or KV layout differ from production can never poison, or be
served from, the production key space.

The namespace is derived from a configuration fingerprint::

    fingerprint = sha256 over
        image id of the service,
        for every bind-mounted regular file: (container target, sha256(content)),
        for every bind-mounted directory: (container target, host path) and,
            for model directories (config.json present), the config's sha256
            and a sorted (name, size) listing of the checkpoint files,
        every service environment variable except an operational denylist
        (capacities, ports, timeouts, logging, profiler, eviction knobs,
        the namespace variables themselves).

A registry file ``NAMESPACE.json`` in the L2 directory maps accepted
fingerprints to the production namespace (``legacy`` = the empty salt that
the entries written before namespaces existed carry). A boot whose
fingerprint is accepted uses the production namespace; any other boot gets
the isolated namespace ``iso-<fingerprint[:16]>``. Acceptance is an operator
decision recorded with ``accept`` after the boot has been shown to produce
the reference fingerprints (same-boot cold_fingerprint IDENTICAL to the
production reference), so a configuration change never silently shares the
production cache.

Subcommands (all take ``--compose-config FILE``, the JSON printed by
``docker compose ... config --format json``, and ``--service NAME``):

  fingerprint   print the fingerprint and the manifest it was computed from
  resolve       print the namespace the boot must use (exit 0), or ``iso-*``
  accept        record the fingerprint as production (``--note`` required);
                the manifest is stored next to the registry
  diff          list the files / env keys that differ between the
                configuration and the stored manifest of an accepted
                fingerprint (``--against FP``, default: every accepted one)
  show          print the registry
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

REGISTRY_NAME = "NAMESPACE.json"
MANIFEST_DIR = "NAMESPACE-manifests"
PRODUCTION_NAMESPACE = "legacy"

# Operational variables that do not change stored bytes or their layout.
ENV_DENYLIST = {
    "LMCACHE_L1_GB", "LMCACHE_L1_INIT_GB", "LMCACHE_L2_GB", "LMCACHE_L2_WORKERS",
    "LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT", "LMCACHE_HTTP_PORT", "LMCACHE_PORT",
    "LMCACHE_LOG", "LMCACHE_HEARTBEAT_INTERVAL", "LMCACHE_WORKER_REAP_TIMEOUT",
    "LMCACHE_START_TIMEOUT", "LMCACHE_L2_EVICTION_POLICY",
    "LMCACHE_L2_EVICTION_WATERMARK", "LMCACHE_L2_EVICTION_RATIO",
    "LMCACHE_WRITE_BACK_ON_EVICT", "LMCACHE_EMERGENCY_EVICT_FOR_PREFETCH",
    "LMCACHE_KV_LOAD_FAILURE_POLICY", "LMCACHE_CACHE_NAMESPACE",
    "LMCACHE_REQUIRE_CACHE_NAMESPACE", "LMCACHE_L2_RESET_ON_START",
    "VLLM_PROFILER_DIR", "VLLM_PROFILER_EXTRA_ARGS", "VLLM_LOGGING_LEVEL",
    "VLLM_LOGGING_CONFIG_PATH", "VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK",
    "HF_HUB_OFFLINE", "HF_HOME", "PYTHONUNBUFFERED", "TERM", "HOME", "PATH",
    "HOSTNAME", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "PORT",
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _image_id(image: str) -> str:
    out = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def build_manifest(compose_config: dict, service: str) -> dict:
    svc = compose_config["services"][service]
    manifest = {"image": svc["image"], "image_id": _image_id(svc["image"]),
                "files": [], "directories": [], "env": {}}
    for vol in svc.get("volumes", []):
        if vol.get("type") != "bind":
            continue
        source, target = vol["source"], vol["target"]
        if os.path.isfile(source):
            manifest["files"].append({"target": target, "sha256": _sha256_file(source)})
        elif os.path.isdir(source):
            # A directory is identified by its container target and host
            # path. Model directories (those carrying a config.json) also
            # contribute the config and a (name, size) listing of their
            # files so a swapped checkpoint changes the fingerprint; scratch
            # and cache directories (/cache, /lmcache/l2) change on their own
            # and contribute the path only.
            entry = {"target": target, "source": source}
            cfg = os.path.join(source, "config.json")
            if os.path.isfile(cfg):
                entry["config_sha256"] = _sha256_file(cfg)
                listing = []
                for name in sorted(os.listdir(source)):
                    p = os.path.join(source, name)
                    if os.path.isfile(p):
                        listing.append([name, os.path.getsize(p)])
                entry["listing_sha256"] = hashlib.sha256(json.dumps(listing).encode()).hexdigest()
            manifest["directories"].append(entry)
        else:
            manifest["files"].append({"target": target, "sha256": "missing"})
    env = svc.get("environment", {})
    if isinstance(env, list):
        env = dict(item.split("=", 1) if "=" in item else (item, "") for item in env)
    for key in sorted(env):
        if key in ENV_DENYLIST:
            continue
        manifest["env"][key] = "" if env[key] is None else str(env[key])
    manifest["files"].sort(key=lambda e: e["target"])
    manifest["directories"].sort(key=lambda e: e["target"])
    return manifest


def fingerprint_of(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def load_registry(l2_dir: str) -> dict:
    path = os.path.join(l2_dir, REGISTRY_NAME)
    if not os.path.exists(path):
        return {"production_namespace": PRODUCTION_NAMESPACE, "accepted": {}}
    with open(path) as fh:
        return json.load(fh)


def save_registry(l2_dir: str, registry: dict) -> None:
    path = os.path.join(l2_dir, REGISTRY_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def manifest_path(l2_dir: str, fp: str) -> str:
    return os.path.join(l2_dir, MANIFEST_DIR, f"{fp}.json")


def diff_manifests(old: dict, new: dict) -> list[str]:
    """Return human-readable lines describing what differs between manifests."""
    lines = []
    if old.get("image_id") != new.get("image_id"):
        lines.append(f"image: {old.get('image')} ({old.get('image_id')}) -> {new.get('image')} ({new.get('image_id')})")
    old_files = {e["target"]: e["sha256"] for e in old.get("files", [])}
    new_files = {e["target"]: e["sha256"] for e in new.get("files", [])}
    for t in sorted(set(old_files) | set(new_files)):
        if t not in old_files:
            lines.append(f"file added:   {t}")
        elif t not in new_files:
            lines.append(f"file removed: {t}")
        elif old_files[t] != new_files[t]:
            lines.append(f"file changed: {t}")
    old_dirs = {e["target"]: e for e in old.get("directories", [])}
    new_dirs = {e["target"]: e for e in new.get("directories", [])}
    for t in sorted(set(old_dirs) | set(new_dirs)):
        if old_dirs.get(t) != new_dirs.get(t):
            lines.append(f"directory changed: {t} ({old_dirs.get(t, {}).get('source')} -> {new_dirs.get(t, {}).get('source')})")
    old_env, new_env = old.get("env", {}), new.get("env", {})
    for k in sorted(set(old_env) | set(new_env)):
        if k not in old_env:
            lines.append(f"env added:   {k}={new_env[k]}")
        elif k not in new_env:
            lines.append(f"env removed: {k}")
        elif old_env[k] != new_env[k]:
            lines.append(f"env changed: {k}: {old_env[k]} -> {new_env[k]}")
    return lines


def resolve(registry: dict, fp: str) -> str:
    if fp in registry.get("accepted", {}):
        return registry.get("production_namespace", PRODUCTION_NAMESPACE)
    return f"iso-{fp[:16]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fingerprint", "resolve", "accept", "diff", "show"])
    ap.add_argument("--compose-config", help="JSON from `docker compose ... config --format json`")
    ap.add_argument("--service", default="kimi-k3-production")
    ap.add_argument("--l2-dir", required=True, help="host path of the LMCache L2 directory holding NAMESPACE.json")
    ap.add_argument("--fingerprint", help="accept: fingerprint to record (default: computed from --compose-config)")
    ap.add_argument("--note", help="accept: evidence that the configuration may share the production namespace")
    ap.add_argument("--manifest-out", help="fingerprint: write the manifest JSON here")
    ap.add_argument("--against", help="diff: accepted fingerprint to compare with (default: all accepted)")
    args = ap.parse_args()

    if args.command == "show":
        print(json.dumps(load_registry(args.l2_dir), indent=2, sort_keys=True))
        return 0

    fp = args.fingerprint
    manifest = None
    if fp is None or args.command in ("fingerprint", "resolve", "diff"):
        if not args.compose_config:
            ap.error("--compose-config is required")
        with open(args.compose_config) as fh:
            compose_config = json.load(fh)
        manifest = build_manifest(compose_config, args.service)
        fp = fingerprint_of(manifest)

    if args.command == "fingerprint":
        if args.manifest_out and manifest is not None:
            with open(args.manifest_out, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
        print(fp)
        return 0

    registry = load_registry(args.l2_dir)
    if args.command == "diff":
        targets = [args.against] if args.against else sorted(registry.get("accepted", {}))
        if not targets:
            print("no accepted fingerprints to compare with")
            return 1
        for other in targets:
            path = manifest_path(args.l2_dir, other)
            if not os.path.exists(path):
                print(f"== {other}: no stored manifest")
                continue
            with open(path) as fh:
                old = json.load(fh)
            lines = diff_manifests(old, manifest)
            print(f"== against {other} ({registry['accepted'][other].get('note', '')[:60]}): {len(lines)} differences")
            for line in lines:
                print("   " + line)
        return 0

    if args.command == "resolve":
        namespace = resolve(registry, fp)
        print(namespace)
        if namespace != registry.get("production_namespace", PRODUCTION_NAMESPACE):
            print(f"# fingerprint {fp} is not accepted; boot is isolated as {namespace}", file=sys.stderr)
        return 0

    if args.command == "accept":
        if not args.note:
            ap.error("--note is required for accept")
        registry.setdefault("accepted", {})[fp] = {
            "accepted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "note": args.note,
            "image": manifest["image"] if manifest else None,
        }
        registry.setdefault("production_namespace", PRODUCTION_NAMESPACE)
        save_registry(args.l2_dir, registry)
        if manifest is not None:
            os.makedirs(os.path.join(args.l2_dir, MANIFEST_DIR), exist_ok=True)
            with open(manifest_path(args.l2_dir, fp), "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
        print(f"accepted {fp} -> {registry['production_namespace']}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
