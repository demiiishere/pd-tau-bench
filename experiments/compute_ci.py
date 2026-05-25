#!/usr/bin/env python3
"""实验 B —— 给 temp=0 单次结果配二项(Wilson)95% 置信区间, 并做配对 McNemar 检验.
纯后处理: 不需要 GPU, 不重跑任何实验, 秒级出结果.

用法:
  python /user/zhujiatong/pd-tau-bench/experiments/compute_ci.py
  python /user/zhujiatong/pd-tau-bench/experiments/compute_ci.py \
      --mcnemar e3c_sft_32b_bon_dpo zero_shot_4b

重要(诚实声明, 也建议照抄进论文):
  Wilson CI 与 McNemar 只刻画"有限测试集抽样"这一个不确定性来源,
  把每个任务的贪心成败当作模型的固定属性. 多种子实验已证明贪心路径
  本身就是高方差抽样 —— 因此这些区间/ p 值是不确定性的【下界】,
  不能据此宣称配置之间的差异稳健.
"""
import argparse
import glob
import json
import math
import os

RESULTS_BASE_DEFAULT = "/user/zhujiatong/pd-tau-bench/data/results"
DOMAINS = ["retail", "airline", "telecom"]
EXPS_DEFAULT = ["zero_shot_4b", "e3a_sft_32b_pd", "e3b_sft_32b_bon",
                "e3c_sft_32b_bon_dpo", "e3d_sft_32b_pd_dpo"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_exp(base, exp):
    """返回 res[domain] = {task_filename: 0/1}"""
    res = {}
    for dom in DOMAINS:
        d = {}
        for f in glob.glob(os.path.join(base, exp, dom, "*_baseline.json")):
            try:
                r = json.load(open(f))["final_reward"]
                d[os.path.basename(f)] = 1 if r == 1.0 else 0
            except Exception:
                pass
        if d:
            res[dom] = d
    return res


def mcnemar_p(b, c):
    """精确双侧 McNemar 检验: b, c 为两类不一致对的个数."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", default=RESULTS_BASE_DEFAULT)
    ap.add_argument("--exps", nargs="+", default=EXPS_DEFAULT)
    ap.add_argument("--mcnemar", nargs=2, metavar=("EXP1", "EXP2"), default=None)
    args = ap.parse_args()

    print("=" * 74)
    print("  temp=0 单次结果  +  Wilson 95% 置信区间")
    print("=" * 74)
    print(f"  {'实验':26s} {'域':9s}  k/n        pass@1    95% CI")
    print("  " + "-" * 70)
    for exp in args.exps:
        res = load_exp(args.results_base, exp)
        if not res:
            print(f"  {exp:26s}  (无结果, 跳过)")
            continue
        tk = tn = 0
        for dom in DOMAINS:
            if dom not in res:
                continue
            vals = list(res[dom].values())
            k, n = sum(vals), len(vals)
            tk += k
            tn += n
            lo, hi = wilson(k, n)
            print(f"  {exp:26s} {dom:9s}  {k:2d}/{n:<3d}     {100*k/n:5.1f}%   "
                  f"[{100*lo:4.1f}%, {100*hi:5.1f}%]")
        lo, hi = wilson(tk, tn)
        print(f"  {exp:26s} {'OVERALL':9s}  {tk:2d}/{tn:<3d}     {100*tk/tn:5.1f}%   "
              f"[{100*lo:4.1f}%, {100*hi:5.1f}%]")
        print("  " + "-" * 70)

    if args.mcnemar:
        e1, e2 = args.mcnemar
        r1, r2 = load_exp(args.results_base, e1), load_exp(args.results_base, e2)
        print()
        print("=" * 74)
        print(f"  配对 McNemar 检验:  {e1}  vs  {e2}")
        print("=" * 74)
        both = e1_only = e2_only = neither = 0
        for dom in DOMAINS:
            d1, d2 = r1.get(dom, {}), r2.get(dom, {})
            for t in set(d1) & set(d2):
                a, b = d1[t], d2[t]
                if a and b:
                    both += 1
                elif a and not b:
                    e1_only += 1
                elif b and not a:
                    e2_only += 1
                else:
                    neither += 1
        print(f"  都成功={both}   仅 {e1} 成功={e1_only}   "
              f"仅 {e2} 成功={e2_only}   都失败={neither}")
        print(f"  McNemar 精确双侧 p = {mcnemar_p(e1_only, e2_only):.4f}")
        print()
        print("  注意: 此 p 值仅考虑测试集抽样, 把每个任务的贪心成败当作固定属性.")
        print("  多种子实验已证明贪心路径本身高方差 —— 故此 p 值高估了把握度,")
        print("  它是不确定性的下界, 不能据此宣称配置间差异稳健.")


if __name__ == "__main__":
    main()
