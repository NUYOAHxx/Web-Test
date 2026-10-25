"""A/B 交替计时：ab_coord.py <out.json> <model: two_arm | synth> <rootA> <rootB> [json_extra] [rounds]
逐目标交替（ABBA），同一时刻只有一个 worker 在算。"""
import json, os, subprocess, sys, time
import numpy as np
out, model, ra, rb = sys.argv[1:5]
extra = sys.argv[5] if len(sys.argv) > 5 else ""
rounds = int(sys.argv[6]) if len(sys.argv) > 6 else 1
PY = sys.executable
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_worker.py")
def start(root):
    p = subprocess.Popen([PY, W, root, model, extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    hello = json.loads(p.stdout.readline())
    return p, hello["sizes"]
pa, sizes = start(ra); pb, _ = start(rb)
def ask(p, mode, k):
    p.stdin.write(f"{mode} {k}\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
res = {"A": [], "B": []}
t0 = time.time(); i = 0
for rnd in range(rounds):
    for mode, n in sizes.items():
        for k in range(n):
            order = (("A", pa), ("B", pb)) if i % 2 == 0 else (("B", pb), ("A", pa))
            for tag, p in order:
                r = ask(p, mode, k); r.update(mode=mode, k=k, rnd=rnd); res[tag].append(r)
            i += 1
for p in (pa, pb):
    p.stdin.write("quit\n"); p.stdin.flush(); p.wait()
json.dump({"model": model, "A": ra, "B": rb, "extra": extra, "res": res, "wall_s": time.time() - t0}, open(out, "w"))
a = np.array([x["ms"] for x in res["A"]]); b = np.array([x["ms"] for x in res["B"]])
same = sum(x["h"] == y["h"] for x, y in zip(res["A"], res["B"]))
print(f"[{model} {extra}] n={len(a)}  identical q {same}/{len(a)}")
for tag, v, src in (("A", a, ra), ("B", b, rb)):
    print(f"  {tag} {os.path.basename(src):18s} mean {v.mean():6.1f}  median {np.median(v):6.1f}  p90 {np.percentile(v, 90):6.1f}  max {v.max():6.1f} ms")
print(f"  B/A: mean {b.mean() / a.mean():.3f}  median-of-ratios {np.median(b / a):.3f}  ratio-of-medians {np.median(b) / np.median(a):.3f}")
