import pickle, sys
a, b = pickle.load(open(sys.argv[1], "rb")), pickle.load(open(sys.argv[2], "rb"))
keys = sorted(set(a) | set(b))
diff = [k for k in keys if a.get(k) != b.get(k)]
print(f"records {len(a)} vs {len(b)}; differing {len(diff)}")
def first_diff(x, y, path=""):
    if type(x) != type(y):
        return f"{path}: type {type(x).__name__} vs {type(y).__name__}"
    if isinstance(x, tuple):
        if len(x) != len(y):
            return f"{path}: len {len(x)} vs {len(y)} :: {str(x)[:300]} || {str(y)[:300]}"
        for i, (u, v) in enumerate(zip(x, y)):
            if u != v:
                return first_diff(u, v, f"{path}[{i}]")
    return f"{path}: {str(x)[:300]} vs {str(y)[:300]}"
for k in diff[: int(sys.argv[3]) if len(sys.argv) > 3 else 12]:
    print(" -", k, first_diff(a.get(k), b.get(k)))
