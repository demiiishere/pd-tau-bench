#!/usr/bin/env python3
"""汇总多种子评估结果: 每个种子的 pass@1, 跨种子 mean±std, 以及 pass@k.

用法:
    python experiments/aggregate_multiseed.py \
        --results-base /user/zhujiatong/pd-tau-bench/data/results/multiseed \
        --label e3c_32b_bondpo \
        --domains retail airline telecom \
        --seeds 0 1 2 3 4 \
        [--compare e3b_32b_bon]

每个种子目录结构: <results-base>/<label>/seed_<S>/<domain>/task_*_baseline.json
"""
import argparse
import json
import math
from pathlib import Path


def load_label(base, label, domains, seeds):
    """返回 res[seed][domain] = [(task_filename, 0/1), ...]"""
    res = {}
    for s in seeds:
        res[s] = {}
        for d in domains:
            ddir = Path(base) / label / f"seed_{s}" / d
            rows = []
            if ddir.is_dir():
                for f in sorted(ddir.glob("*_baseline.json")):
                    try:
                        r = json.load(open(f))["final_reward"]
                        rows.append((f.name, 1 if r == 1.0 else 0))
                    except Exception:
                        pass
            res[s][d] = rows
    return res


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v)


def per_seed_overall(res, seeds, domains):
    """每个种子的整体 pass@1 (按测试任务数自然加权 = 合并池化)"""
    out = {}
    for s in seeds:
        succ = sum(sum(r for _, r in res[s][d]) for d in domains)
        tot = sum(len(res[s][d]) for d in domains)
        out[s] = (succ / tot) if tot else float("nan")
    return out


def summarize(label, res, seeds, domains):
    print(f"\n{'=' * 68}")
    print(f"  {label}")
    print(f"{'=' * 68}")
    print("  domain   " + "".join(f"  seed{s}" for s in seeds) + "     mean±std")
    for d in domains:
        vals, cells = [], ""
        for s in seeds:
            rows = res[s][d]
            p = (sum(r for _, r in rows) / len(rows)) if rows else float("nan")
            vals.append(p)
            cells += f"  {p * 100:5.1f}" if not math.isnan(p) else "    -- "
        m, sd = mean_std(vals)
        print(f"  {d:9s}{cells}     {m * 100:5.1f} ± {sd * 100:4.1f}")
    ov = per_seed_overall(res, seeds, domains)
    cells = "".join(
        f"  {ov[s] * 100:5.1f}" if not math.isnan(ov[s]) else "    -- " for s in seeds
    )
    m, sd = mean_std(list(ov.values()))
    print(f"  {'OVERALL':9s}{cells}     {m * 100:5.1f} ± {sd * 100:4.1f}")

    # pass@k: 按 (domain, task) 跨种子聚合
    task_succ = {}
    for s in seeds:
        for d in domains:
            for fn, r in res[s][d]:
                task_succ.setdefault((d, fn), []).append(r)
    n = len(task_succ)
    if n:
        any_s = sum(1 for v in task_succ.values() if any(v))
        all_s = sum(1 for v in task_succ.values() if v and all(v))
        print(f"  任务数={n}   任一种子成功(pass@k)={any_s}/{n}={any_s / n * 100:.1f}%"
              f"   全种子成功={all_s}/{n}={all_s / n * 100:.1f}%")
    return ov


def sign_test(diffs):
    """双侧符号检验 p 值, 仅用标准库. 注意 n=5 时最小可达 p=0.0625."""
    from math import comb
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--domains", nargs="+", default=["retail", "airline", "telecom"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--compare", default=None, help="另一个 label, 与 --label 做配对比较")
    args = ap.parse_args()

    res_a = load_label(args.results_base, args.label, args.domains, args.seeds)
    ov_a = summarize(args.label, res_a, args.seeds, args.domains)

    if args.compare:
        res_b = load_label(args.results_base, args.compare, args.domains, args.seeds)
        ov_b = summarize(args.compare, res_b, args.seeds, args.domains)

        print(f"\n{'=' * 68}")
        print(f"  配对比较:  {args.label}  −  {args.compare}")
        print(f"{'=' * 68}")
        diffs = []
        for s in args.seeds:
            a, b = ov_a.get(s, float("nan")), ov_b.get(s, float("nan"))
            if math.isnan(a) or math.isnan(b):
                continue
            diffs.append(a - b)
            print(f"  seed {s}:  {a * 100:5.1f}  vs  {b * 100:5.1f}    diff = {(a - b) * 100:+5.1f} pp")
        if diffs:
            m, sd = mean_std(diffs)
            print(f"  平均差 = {m * 100:+.2f} pp ± {sd * 100:.2f}   (n={len(diffs)})")
            print(f"  符号检验  p = {sign_test(diffs):.4f}")
            try:
                from scipy import stats
                a = [ov_a[s] for s in args.seeds if s in ov_a and not math.isnan(ov_a[s])
                     and s in ov_b and not math.isnan(ov_b[s])]
                b = [ov_b[s] for s in args.seeds if s in ov_a and not math.isnan(ov_a[s])
                     and s in ov_b and not math.isnan(ov_b[s])]
                t = stats.ttest_rel(a, b)
                w = stats.wilcoxon(a, b) if len(a) >= 1 and any(x != y for x, y in zip(a, b)) else None
                print(f"  配对 t 检验  p = {t.pvalue:.4f}")
                if w is not None:
                    print(f"  Wilcoxon   p = {w.pvalue:.4f}")
            except ImportError:
                print("  (未装 scipy: 仅给出符号检验; n=5 时符号检验最小 p=0.0625,"
                      " 如需 p<0.05 请装 scipy 或加到 >=6 个种子)")


if __name__ == "__main__":
    main()
