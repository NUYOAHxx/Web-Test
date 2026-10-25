import pickle, numpy as np
import sys, os
D = os.path.join(sys.argv[1], "")
modes = ("full", "tool_dir", "tool_cone")
for model in ("two_arm", "wide"):
    per = {}
    for ver in ("v13", "v14"):
        rows_r = []
        for r in (1, 2):
            res = pickle.load(open(f"{D}{ver}_{model}_r{r}.pkl", "rb"))
            rows_r.append([x for m in modes for x in res[m]])
        per[ver] = rows_r
    # 结果一致性（两版同一轮、同一目标）
    same = all(
        (a["q"] is None and b["q"] is None) or (a["q"] is not None and b["q"] is not None and np.array_equal(a["q"], b["q"]))
        for r in range(2) for a, b in zip(per["v13"][r], per["v14"][r])
    )
    print(f"[{model}] q identical v13 vs v14 (both rounds): {same}")
    for ver in ("v13", "v14"):
        ms_r = [np.array([x["ms"] for x in rows]) for rows in per[ver]]
        ms = np.concatenate(ms_r)
        solved = sum(x["q"] is not None for x in per[ver][0])
        print(f"  {ver} solved {solved}/{len(per[ver][0])}  mean {ms.mean():.1f}  median {np.median(ms):.1f}  "
              f"p90 {np.percentile(ms, 90):.1f}  max {ms.max():.0f}  rounds mean " + " / ".join(f"{m.mean():.1f}" for m in ms_r))
    a = np.concatenate([np.array([x["ms"] for x in rows]) for rows in per["v13"]])
    b = np.concatenate([np.array([x["ms"] for x in rows]) for rows in per["v14"]])
    ratio = b / a
    print(f"  paired ratio v14/v13: median {np.median(ratio):.3f}  mean-of-means {b.mean() / a.mean():.3f}")
