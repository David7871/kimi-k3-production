"""Decode-only closed-loop bench: tiny prompt, long ignore_eos generations, streaming; counts
only tokens arriving inside [WARM, WARM+WIN] so prefill/ramp are excluded.
Usage: decode_bench.py PORT CONC[,CONC...] [WIN_S=20] [WARM_S=5] [MAX_TOKENS=2048]
Prints one line per concurrency: C<n> decode=<agg tok/s> per_req=<tok/s>"""
import asyncio
import json
import sys
import time

import httpx

port = int(sys.argv[1])
concs = [int(x) for x in sys.argv[2].split(",")]
win = float(sys.argv[3]) if len(sys.argv) > 3 else 20
warm = float(sys.argv[4]) if len(sys.argv) > 4 else 5
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
URL = f"http://{__import__("os").environ.get("HOST","127.0.0.1")}:{port}/v1/completions"


async def worker(cl, t_lo, t_hi, counts):
    while time.time() < t_hi:
        body = {"model": "Kimi-K3-QSRT-K2", "prompt": "Count upward forever:", "max_tokens": max_tokens,
                "temperature": 0.7, "ignore_eos": True, "stream": True}
        async with cl.stream("POST", URL, json=body, timeout=600) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.time()
                if now >= t_hi:
                    return
                if now >= t_lo and json.loads(line[6:])["choices"][0].get("text"):
                    counts[0] += 1


async def run(conc):
    counts = [0]
    t = time.time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=conc + 4)) as cl:
        await asyncio.gather(*[worker(cl, t + warm, t + warm + win, counts) for _ in range(conc)])
    agg = counts[0] / win
    print(f"C{conc} decode={agg:.1f} per_req={agg / conc:.1f}", flush=True)


for c in concs:
    asyncio.run(run(c))
