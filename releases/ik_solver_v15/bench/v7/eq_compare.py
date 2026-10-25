import sys, pickle, numpy as np
a = pickle.load(open(sys.argv[1], "rb")); b = pickle.load(open(sys.argv[2], "rb"))
def same(x, y, path, diffs):
    if isinstance(x, dict):
        if set(x) != set(y): diffs.append((path, "keys", sorted(set(x) ^ set(y)))); return
        for k in x: same(x[k], y[k], path + "." + str(k), diffs)
    elif isinstance(x, (list, tuple)):
        if len(x) != len(y): diffs.append((path, "len", len(x), len(y))); return
        for i, (u, v) in enumerate(zip(x, y)): same(u, v, f"{path}[{i}]", diffs)
    elif isinstance(x, np.ndarray):
        if not isinstance(y, np.ndarray) or x.shape != y.shape or x.tobytes() != y.tobytes():
            diffs.append((path, "array", None if not isinstance(y, np.ndarray) else float(np.max(np.abs(x - y)))))
    elif isinstance(x, float) and isinstance(y, float):
        if not (x == y or (x != x and y != y)) or (x == 0 and np.copysign(1, x) != np.copysign(1, y)):
            diffs.append((path, "float", x, y))
    else:
        if x != y and not (x is None and y is None): diffs.append((path, "value", x, y))
tot = 0
for case in a:
    diffs = []
    same(a[case], b[case], case, diffs)
    tot += len(diffs)
    nsolve = len(a[case]); niters = sum(sum(p.get("iters", 0) for p in r["per_seed"]) for r in a[case])
    print(f"{case:14s} solves {nsolve:4d} DLS-iters {niters:6d} diffs {len(diffs)}", diffs[:3])
print("TOTAL DIFFS", tot)
