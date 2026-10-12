#!/bin/bash
# runs the benchmark scenario groups in separate processes (keeps memory bounded) and merges the results
cd "$(dirname "$0")/.."
N=${N:-20}
mkdir -p results/parts
for g in m1 m1hard m2 auto specific torque m3 box near; do
  PYTHONPATH=. python tests/run_benchmark.py --n $N --only $g --out results/parts/$g.json > results/parts/$g.log 2>&1
  echo "$g done exit=$? $(date +%T)"
done
python - << 'PY'
import json, glob
out = None
for g in ["m1", "m1hard", "m2", "auto", "specific", "torque", "m3", "box", "near"]:
    d = json.load(open(f"results/parts/{g}.json"))
    if out is None:
        out = {k: v for k, v in d.items() if k != "rows"}; out["rows"] = []; out["parts"] = {}
    out["rows"] += d["rows"]; out["parts"][g] = {k: d[k] for k in ("init_time", "warmup_time", "nlp_builds", "nlp_build_time")}
json.dump(out, open("results/benchmark.json", "w"), indent=1)
print("merged", len(out["rows"]))
PY
