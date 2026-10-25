"""配对质量比较：quality.py sp_v6.pkl sp_plan2.pkl sp_v7.pkl -> 打印 JSON 汇总"""
import sys, pickle, json
import numpy as np
qr = np.array([.05,.1,.25,-.6,.4,-.8,.3,.6,-.2])
base = pickle.load(open(sys.argv[1], "rb"))
out = {}
for path in sys.argv[2:]:
    other = pickle.load(open(path, "rb")); tag = path.split("_",1)[1].rsplit(".",1)[0]
    for case in base:
        A, B = base[case], other[case]; n = len(A)
        okA = [a["q"] is not None for a in A]; okB = [b["q"] is not None for b in B]
        both = [i for i in range(n) if okA[i] and okB[i]]
        pa = np.array([A[i]["peak"] for i in both]); pb = np.array([B[i]["peak"] for i in both])
        dq_ready = lambda R: np.sqrt(np.mean([np.mean((R[i]["q"]-qr)**2) for i in both]))
        waist = lambda R: float(np.mean([np.max(np.abs(R[i]["q"][:2]-qr[:2])) for i in both]))
        jump = lambda R: np.array([np.max(np.abs(R[i]["q"]-R[i]["ref_prev"])) for i in both])
        same = sum(1 for i in both if np.array_equal(A[i]["q"], B[i]["q"]))
        close = sum(1 for i in both if np.max(np.abs(A[i]["q"]-B[i]["q"])) < 1e-3)
        diff_sol = [float(np.max(np.abs(A[i]["q"]-B[i]["q"]))) for i in both]
        ms_a = np.array([a["ms"] for a in A]); ms_b = np.array([b["ms"] for b in B])
        out.setdefault(case, {})["v6"] = dict(succ=sum(okA), peak_mean=round(float(pa.mean()),4), peak_max=round(float(pa.max()),4),
            q_ready_rms=round(float(dq_ready(A)),4), waist_dev=round(waist(A),4), jump_max=round(float(jump(A).max()),3),
            jump_gt_pi=int((jump(A)>np.pi).sum()), ms_med=round(float(np.median(ms_a)),2), ms_p90=round(float(np.percentile(ms_a,90)),2),
            ms_mean=round(float(ms_a.mean()),2), seeds=round(float(np.mean([a["seeds"] for a in A])),3), iters=round(float(np.mean([a["iters"] for a in A])),1))
        out[case][tag] = dict(succ=sum(okB), peak_mean=round(float(pb.mean()),4), peak_max=round(float(pb.max()),4),
            q_ready_rms=round(float(dq_ready(B)),4), waist_dev=round(waist(B),4), jump_max=round(float(jump(B).max()),3),
            jump_gt_pi=int((jump(B)>np.pi).sum()), ms_med=round(float(np.median(ms_b)),2), ms_p90=round(float(np.percentile(ms_b,90)),2),
            ms_mean=round(float(ms_b.mean()),2), seeds=round(float(np.mean([b["seeds"] for b in B])),3), iters=round(float(np.mean([b["iters"] for b in B])),1),
            speedup_mean=round(float(ms_a.mean()/ms_b.mean()),2), speedup_med=round(float(np.median(ms_a)/np.median(ms_b)),2),
            speedup_pair_med=round(float(np.median(ms_a/ms_b)),2),
            same_as_v6=f"{same}/{len(both)}", close_1e3=f"{close}/{len(both)}", sol_diff_max=round(max(diff_sol),4),
            peak_diff_mean=round(float((pb-pa).mean()),5), peak_diff_max=round(float((pb-pa).max()),5), peak_worse_gt_0_01=int(((pb-pa)>0.01).sum()),
            skipped=sum(1 for b in B if b["skipped"]))
print(json.dumps(out, ensure_ascii=False, indent=1))
