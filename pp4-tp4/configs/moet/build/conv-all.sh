#!/bin/bash
for spec in "1-23 w1" "24-46 w2" "47-69 w3" "70-92 w4"; do
  set -- $spec
  docker run --rm --network none --cpus 24 --entrypoint /opt/venv/bin/python     -v /data/models:/data/models -v /data/models/k3-moet-work:/work     -e CONVERT_THREADS=24 kimi-k3-up-ray:local     /work/convert_moet_packs.py --layers $1 --golden 4     --out /data/models/k3-moet-packs > /data/models/k3-moet-work/conv-$2.log 2>&1 &
done
wait
echo ALL-WORKERS-DONE
