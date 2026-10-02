#!/usr/bin/env python3
"""GSM8K-200 quality probe for a live K3 lane (text-only chat completions).

Usage: k3_gsm8k.py PORT N OUT.jsonl [CONC]
Greedy (temperature=0), max_tokens 4096 (K3 reasoning headroom), fixed
prompt. Per-item correctness logged to OUT.jsonl; final line prints
"GSM8K acc=<float> n=<int> correct=<int>".
"""
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = "/data/models/llm-inference-bench/data/gsm8k_test.jsonl"
PROMPT = ("Solve the following grade-school math problem. Reason step by "
          "step, then finish with exactly one line: 'The answer is <number>'."
          "\n\nProblem: {q}")
ANS_RE = re.compile(r"answer is[^0-9\-]*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
NUM_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def gold_of(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def extract(text):
    m = ANS_RE.search(text)
    cand = m.group(1) if m else None
    if cand is None:
        nums = NUM_RE.findall(text)
        cand = nums[-1] if nums else ""
    return cand.replace(",", "")


def one(port, item):
    q = item["question"]
    body = {
        "model": "Kimi-K3-QSRT-K2",
        "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
        "temperature": 0,
        "max_tokens": 4096,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    toks, finish = None, None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        finish = d["choices"][0].get("finish_reason")
        toks = (d.get("usage") or {}).get("completion_tokens")
        text = (msg.get("content") or "")
        if msg.get("reasoning_content"):
            text = msg["reasoning_content"] + "\n" + text
        pred = extract(text)
    except Exception as e:  # noqa: BLE001 - per-item isolation
        text, pred = f"ERROR: {e}", ""
    gold = gold_of(item["answer"])
    ok = False
    try:
        ok = abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        pass
    return {"question": q[:80], "gold": gold, "pred": pred, "ok": ok,
            "completion_tokens": toks, "finish_reason": finish,
            "tail": text[-200:]}


def main():
    port, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    conc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    items = []
    with open(DATA) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    with ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(lambda it: one(port, it), items))
    correct = sum(r["ok"] for r in results)
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    toks = sorted(r["completion_tokens"] for r in results
                  if r["completion_tokens"] is not None)
    cut = sum(r["finish_reason"] == "length" for r in results)
    tok_s = (f" tokens_sum={sum(toks)} tokens_mean={sum(toks) / len(toks):.1f}"
             f" tokens_median={toks[len(toks) // 2]}" if toks else
             " tokens=unavailable")
    print(f"GSM8K acc={correct / len(results):.4f} n={len(results)} "
          f"correct={correct}{tok_s} truncated={cut}")


if __name__ == "__main__":
    main()
