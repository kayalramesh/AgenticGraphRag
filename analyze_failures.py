"""
python analyze_failures.py            # reads eval_results_{rag,graphrag,agentic}.jsonl
Prints every wrong answer with gold vs predicted, grouped by failure type, and a cross-pipeline table
so you can see which questions NO pipeline solves (those need a prompt/model/retrieval fix, not more agent steps).
"""
import json
from collections import defaultdict

NAMES = ["rag", "graphrag", "agentic"]
data = {}
for n in NAMES:
    try:
        data[n] = [json.loads(l) for l in open(f"eval_results_{n}.jsonl", encoding="utf-8")]
    except FileNotFoundError:
        pass

for n, rows in data.items():
    print(f"\n{'=' * 20} {n}: wrong answers {'=' * 20}")
    for i, r in enumerate(rows):
        if r["correct"]:
            continue
        kind = "GENERATION (gold was in context)" if r["gold_in_context"] else "RETRIEVAL (gold never reached LLM)"
        print(f"\n#{i} [{kind}]\n  Q:    {r['question']}\n  gold: {r['gold']}\n  pred: {str(r['answer'])[:200]}")
        if n == "agentic":
            print("  path: " + " > ".join(s["action"] for s in r.get("steps", [])), "| stop:", r.get("stop_reason"))

print(f"\n{'=' * 20} per-question matrix (1 = correct) {'=' * 20}")
if data:
    n_rows = min(len(v) for v in data.values())
    unsolved = []
    for i in range(n_rows):
        flags = {n: int(data[n][i]["correct"]) for n in data}
        if not any(flags.values()):
            unsolved.append(i)
        print(i, flags, data[NAMES[0] if NAMES[0] in data else list(data)[0]][i]["question"][:70])
    print("\nSolved by NO pipeline:", unsolved)
