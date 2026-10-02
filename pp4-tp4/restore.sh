#!/usr/bin/env bash
# Restore the committed K3 PP4xTP4 lane snapshot onto rtx-01 (head) and rtx-02 (worker).
# Run from the Mac. The live files it replaces are first moved to <name>.pre-restore-<ts>.
# Usage: bash restore.sh [--dry-run]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=/data/models/k3-run
TS=$(date +%Y%m%d-%H%M%S)
DRY=${1:-}
rs() { rsync -a ${DRY:+--dry-run -v} "$@"; }

# 1. stage the snapshot on rtx-01
rs --delete "$HERE/run/" rtx-01:/tmp/k3-lane-restore/
[ -n "$DRY" ] && { echo "dry run: staged list above, nothing changed on hosts"; exit 0; }

# 2. back up live trees, install, and mirror to rtx-02 over the direct link
ssh rtx-01 "set -e
  for d in overlay-vllm-l3 overlay-b12x-l3; do cp -a $RUN/\$d $RUN/\$d.pre-restore-$TS; done
  rsync -a /tmp/k3-lane-restore/overlay-vllm-l3/ $RUN/overlay-vllm-l3/
  rsync -a /tmp/k3-lane-restore/overlay-b12x-l3/ $RUN/overlay-b12x-l3/
  for f in /tmp/k3-lane-restore/scripts/* /tmp/k3-lane-restore/env/* /tmp/k3-lane-restore/overlays-mb/*; do
    sub=\$(basename \$(dirname \$f)); [ -e $RUN/\$sub/\$(basename \$f) ] && cp -a $RUN/\$sub/\$(basename \$f) $RUN/\$sub/\$(basename \$f).pre-restore-$TS
    cp -a \$f $RUN/\$sub/
  done
  for d in overlay-vllm-l3 overlay-b12x-l3 scripts env overlays-mb; do rsync -a $RUN/\$d/ 10.10.10.2:$RUN/\$d/; done"

# 3. bench env string used by bench/*.sh
scp -q "$HERE/bench/stack2.envstr" rtx-01:stack2.envstr

# 4. verify against the manifest on both hosts
M=$(sed 's#^\([0-9a-f]*\)  run/#\1  #' "$HERE/MANIFEST.md5")
ssh rtx-01 "cd $RUN && md5sum --quiet -c -" <<<"$M" && echo "verified rtx-01"
ssh rtx-01 "ssh 10.10.10.2 'cd $RUN && md5sum --quiet -c -'" <<<"$M" && echo "verified rtx-02"
echo "image expected: $(cat "$HERE/IMAGE.txt")"
ssh rtx-01 'docker image inspect kimi-k3-up-ray:local --format "image present:  {{.Id}} {{.Created}}"'
