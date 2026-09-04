#!/usr/bin/env python3
"""Open the island reduce-scatter all-reduce to TP8 (two islands).

The equal-quarter island_rs runtime was written for the TP16 rigs (four
4-GPU islands). The partition is LANE-based (each rank owns a fixed quarter
of the vector regardless of island count), islands are ``divmod(rank, 4)``,
and the launcher is rank-specialized with world_size folded at trace time —
so a two-island TP8 world is structurally representable. This host stack is
a 2-tier PEX88096 cascade whose physical switch boundary is exactly ranks
{0-3} / {4-7}, so island_rs minimizes crossings of the single x16 Gen4
inter-switch link that all-peer oneshot hammers 4x4 per all-reduce.

Changes:
- pcie_island_rs.SUPPORTED_WORLD_SIZES: (16,) -> (8, 16). The dispatcher's
  ISLAND_RS_WORLD_SIZES is an alias import, and _maybe_island_rs already
  requires the explicit B12X_PCIE_ALLREDUCE_ALGORITHM=island_rs opt-in.
- _island_rs_cute.island_rs_peers / _IslandRSLaunch: accept world 8, derive
  num_islands = world // 4.
- The three cross-island loops iterate a compile-time _MAX_ISLANDS=4; guard
  them with ``j < num_islands`` const-exprs. Without the guard the combine
  loop would SUM placeholder slabs at TP8. Island-local loops (RS/AG) are
  island-size-based and unchanged.
"""

import os
from pathlib import Path

HOST = Path(
    os.environ.get(
        "K3PATCH_ISLAND_HOST",
        "/opt/infernal-invocation/b12x/b12x/comm/pcie/pcie_island_rs.py",
    )
)
CUTE = Path(
    os.environ.get(
        "K3PATCH_ISLAND_CUTE",
        "/opt/infernal-invocation/b12x/b12x/comm/pcie/_island_rs_cute.py",
    )
)
MARKER = "kimi-k3-island-rs-ws8"

HOST_OLD = "SUPPORTED_WORLD_SIZES = (16,)\n"
HOST_NEW = (
    "# kimi-k3-island-rs-ws8: two 4-GPU islands map onto the TP8 cascade\n"
    "# (PEX88096 switch boundary = ranks 0-3 / 4-7).\n"
    "SUPPORTED_WORLD_SIZES = (8, 16)\n"
)

PEERS_OLD = '''    if world_size != _MAX_WORLD_SIZE:
        raise ValueError(f"island reduce-scatter requires TP16, got TP{world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for TP{world_size}")
    island, lane = divmod(rank, _ISLAND_SIZE)
    lanes = {island * _ISLAND_SIZE + p for p in range(_ISLAND_SIZE)}
    partners = {j * _ISLAND_SIZE + lane for j in range(_MAX_ISLANDS)}
'''

PEERS_NEW = '''    # kimi-k3-island-rs-ws8: any whole-island world up to _MAX_WORLD_SIZE.
    if world_size % _ISLAND_SIZE or not 0 < world_size <= _MAX_WORLD_SIZE:
        raise ValueError(
            f"island reduce-scatter requires a whole-island world, got TP{world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for TP{world_size}")
    island, lane = divmod(rank, _ISLAND_SIZE)
    num_islands = world_size // _ISLAND_SIZE
    lanes = {island * _ISLAND_SIZE + p for p in range(_ISLAND_SIZE)}
    partners = {j * _ISLAND_SIZE + lane for j in range(num_islands)}
'''

INIT_OLD = '''        if world_size != _MAX_WORLD_SIZE:
            raise ValueError(f"unsupported world size {world_size}")
'''

INIT_NEW = '''        # kimi-k3-island-rs-ws8: two-island TP8 or four-island TP16.
        if world_size not in (8, _MAX_WORLD_SIZE):
            raise ValueError(f"unsupported world size {world_size}")
'''

LANE_OLD = '''        self._island = self._rank // _ISLAND_SIZE
        self._lane = self._rank % _ISLAND_SIZE
'''

LANE_NEW = '''        self._island = self._rank // _ISLAND_SIZE
        self._lane = self._rank % _ISLAND_SIZE
        self._num_islands = self._world_size // _ISLAND_SIZE
'''

XS_STORE_OLD = '''            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    partner = j * _ISLAND_SIZE + self._lane
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[partner].toint()),
                            _XS_ARRIVED,
'''

XS_STORE_NEW = '''            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island and j < self._num_islands):
                    partner = j * _ISLAND_SIZE + self._lane
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[partner].toint()),
                            _XS_ARRIVED,
'''

XS_WAIT_OLD = '''            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    _wait_for(
                        _flag_address(self_base, _XS_ARRIVED, bidx, Int32(j)),
'''

XS_WAIT_NEW = '''            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island and j < self._num_islands):
                    _wait_for(
                        _flag_address(self_base, _XS_ARRIVED, bidx, Int32(j)),
'''

COMBINE_OLD = '''            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                partner = j * _ISLAND_SIZE + self._lane
                partner_part = cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        Int64(slabs[partner].toint()) + part_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ).align(4),
                    dtype=Uint32,
                )
                lo, hi = unpack_bf16x2(partner_part[index])
                total_lo += lo
                total_hi += hi
'''

COMBINE_NEW = '''            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j < self._num_islands):
                    partner = j * _ISLAND_SIZE + self._lane
                    partner_part = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[partner].toint()) + part_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                    lo, hi = unpack_bf16x2(partner_part[index])
                    total_lo += lo
                    total_hi += hi
'''


def apply(path: Path, text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {what} anchor in {path}, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    host = HOST.read_text()
    if MARKER in host:
        print(f"pcie_island_rs already patched: {MARKER}")
    else:
        host = apply(HOST, host, HOST_OLD, HOST_NEW, "world-sizes")
        compile(host, str(HOST), "exec")
        HOST.write_text(host)
        print(f"pcie_island_rs patched: {MARKER}")

    cute = CUTE.read_text()
    if MARKER in cute:
        print(f"_island_rs_cute already patched: {MARKER}")
        return
    cute = "# " + MARKER + "\n" + cute
    cute = apply(CUTE, cute, PEERS_OLD, PEERS_NEW, "peers")
    cute = apply(CUTE, cute, INIT_OLD, INIT_NEW, "init-check")
    cute = apply(CUTE, cute, LANE_OLD, LANE_NEW, "lane-attrs")
    cute = apply(CUTE, cute, XS_STORE_OLD, XS_STORE_NEW, "xs-store")
    cute = apply(CUTE, cute, XS_WAIT_OLD, XS_WAIT_NEW, "xs-wait")
    cute = apply(CUTE, cute, COMBINE_OLD, COMBINE_NEW, "combine")
    compile(cute, str(CUTE), "exec")
    CUTE.write_text(cute)
    print(f"_island_rs_cute patched: {MARKER} (peers, init, 3 island loops)")


if __name__ == "__main__":
    main()
