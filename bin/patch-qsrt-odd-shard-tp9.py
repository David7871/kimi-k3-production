#!/usr/bin/env python3
"""Enable odd TP shard counts for the pure-K2 QSRT atom partition.

The packaged ``_balanced_pure_k2_atom_partition`` requires an even shard
count because it assigns ``shard_count // 2`` ranks to each of the two
transformed preactivation halves. An odd count (TP=9) cannot split the
halves symmetrically, so assign ``ceil(n / 2)`` ranks to the first half and
``floor(n / 2)`` to the second. Every extent still covers whole H128 atom
blocks inside one half; ranks whose source extent is smaller than the padded
runtime width rely on the loader's zero-fill tail exactly like the even
path's remainder ranks (B12X virtual-TP contract).

TP=9 layout produced (first_atom_slot, atom_count):
  rank0 (0,12) rank1 (12,12) rank2 (24,8) rank3 (32,8) rank4 (40,8)
  rank5 (48,12) rank6 (60,12) rank7 (72,12) rank8 (84,12)   sum=96
All extents are multiples of 4 atoms and never cross the half boundary at 48.
Even shard counts keep byte-identical behavior ((n+1)//2 == n//2).
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_QSRT_ATOMS_V2",
        "/opt/infernal-invocation/vllm/vllm/model_executor/layers/"
        "quantization/kquant_qsrt_atoms_v2.py",
    )
)
MARKER = "odd-shard-pure-k2-partition"

OLD = '''    if shard_count == 1:
        return 0, ATOM_SLOTS
    if shard_count < 1 or shard_count > ATOM_SLOTS or shard_count % 2:
        raise ValueError("pure-K2 atom sharding requires an even shard count")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in 0..{shard_count - 1}")
    ranks_per_half = shard_count // 2
    half = shard_index // ranks_per_half
    rank_in_half = shard_index % ranks_per_half
'''

NEW = '''    if shard_count == 1:
        return 0, ATOM_SLOTS
    # odd-shard-pure-k2-partition: an odd shard count cannot split the two
    # transformed preactivation halves symmetrically. Assign ceil(n/2) ranks
    # to the first half and floor(n/2) to the second. Every extent still
    # covers whole H128 blocks inside one half; ranks whose source extent is
    # smaller than the padded runtime width rely on the loader zero-fill
    # tail, exactly like the even path's remainder ranks.
    if shard_count < 1 or shard_count > ATOM_SLOTS:
        raise ValueError("pure-K2 atom sharding requires 1..96 shards")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in 0..{shard_count - 1}")
    first_half_ranks = (shard_count + 1) // 2
    if shard_index < first_half_ranks:
        half = 0
        rank_in_half = shard_index
        ranks_per_half = first_half_ranks
    else:
        half = 1
        rank_in_half = shard_index - first_half_ranks
        ranks_per_half = shard_count - first_half_ranks
'''


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"atoms_v2 already patched: {MARKER}")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one partition anchor in {PATH}, found {count}"
        )
    text = text.replace(OLD, NEW, 1)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"atoms_v2 patched: {MARKER}")

    # Self-check: TP=9 must produce a full, non-overlapping, block-aligned
    # cover of the 96-slot axis with no half-crossing extent.
    namespace: dict = {}
    exec(  # noqa: S102 - executing the just-patched trusted source
        compile(text, str(PATH), "exec"), namespace
    )
    partition = namespace["_balanced_pure_k2_atom_partition"]
    for shards in (2, 4, 8, 9, 12, 16):
        extents = [partition(shards, i) for i in range(shards)]
        covered: list[int] = []
        for first, rows in extents:
            if first % 4 or rows % 4:
                raise SystemExit(f"unaligned extent at shards={shards}: {extents}")
            if first < 48 < first + rows:
                raise SystemExit(f"half-crossing extent at shards={shards}: {extents}")
            covered.extend(range(first, first + rows))
        if sorted(covered) != list(range(96)):
            raise SystemExit(f"non-covering partition at shards={shards}: {extents}")
    print("atoms_v2 partition self-check passed for shards 2,4,8,9,12,16")


if __name__ == "__main__":
    main()
