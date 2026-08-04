# -*- coding: utf-8 -*-
"""因子试验登记册 + 多重检验校正

为什么必须记录试过的因子数量：
  单次试验里 t>2 (p<0.05) 看似显著，但如果你试了20个因子，纯噪声下**期望就有1个**
  达到 p<0.05。不记录试验次数，你迟早会把运气当成alpha。

本模块提供三个校正基准(从松到严)：
  1. Bonferroni: 门槛 t_crit = Z^{-1}(1 - α/(2N))，最保守但过度惩罚(假设试验独立)
  2. Bailey & López de Prado 期望最大t: 零假设下N次试验中最大t的期望值
     E[max z] ≈ (1-γ)Z^{-1}(1-1/N) + γZ^{-1}(1-1/(N·e))，γ=欧拉常数
     → 你的最佳因子t必须**显著超过**这个值才有意义
  3. Harvey-Liu-Zhu(2016)经验门槛: 金融因子研究普遍建议 t>3.0
     (他们统计了数百篇论文的因子，认为t=2的旧标准已不足)

⚠登记册只统计"被评估过的因子配置"。真实的试验次数还包括你调参数、
  换股票池、换持有期的所有尝试——诚实一点，宁可多记不要少记。
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from statistics import NormalDist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "factor_trials.csv")

_FIELDS = ["试验时间", "因子名", "股票池", "持有期", "调仓", "起止", "样本期数",
           "IC均值", "ICIR年化", "t统计量", "多空年化", "多空夏普", "备注"]

_EULER_GAMMA = 0.5772156649015329


def log_trial(name: str, stats: dict, universe: str = "", horizon: int = 0,
              rebalance: int = 0, period: str = "", ls_ann: float = float("nan"),
              ls_sharpe: float = float("nan"), note: str = "") -> None:
    """记一次因子试验(evaluate_factor会自动调用)"""
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({
            "试验时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "因子名": name, "股票池": universe, "持有期": horizon, "调仓": rebalance,
            "起止": period, "样本期数": stats.get("样本期数", ""),
            "IC均值": round(stats.get("IC均值", float("nan")), 5),
            "ICIR年化": round(stats.get("ICIR(年化)", float("nan")), 4),
            "t统计量": round(stats.get("t统计量", float("nan")), 3),
            "多空年化": round(ls_ann, 4) if ls_ann == ls_ann else "",
            "多空夏普": round(ls_sharpe, 3) if ls_sharpe == ls_sharpe else "",
            "备注": note,
        })


def load_trials() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def expected_max_t(n_trials: int) -> float:
    """零假设下N次独立试验中最大t的**期望值** (Bailey & López de Prado近似)。
    ⚠这是"噪声地板"不是临界值：纯噪声试N次，典型的最好成绩就是这个数。
    你的因子t只是勉强超过它 → 大概率是运气。真正的门槛用 bonferroni_t()。"""
    if n_trials < 2:
        return 0.0   # 单次抽样的期望最大值就是均值0
    nd = NormalDist()
    return ((1 - _EULER_GAMMA) * nd.inv_cdf(1 - 1 / n_trials)
            + _EULER_GAMMA * nd.inv_cdf(1 - 1 / (n_trials * math.e)))


def bonferroni_t(n_trials: int, alpha: float = 0.05) -> float:
    """Bonferroni校正后的双尾临界t值"""
    return NormalDist().inv_cdf(1 - alpha / (2 * max(n_trials, 1)))


def report(verbose: bool = True) -> dict:
    """汇总所有试验并给出多重检验校正后的判定"""
    trials = load_trials()
    n = len(trials)
    if n == 0:
        print("登记册为空——还没有记录任何因子试验")
        return {}

    def _t(row):
        try:
            return abs(float(row["t统计量"]))
        except (ValueError, TypeError):
            return float("nan")

    n_unique = len({r["因子名"] for r in trials})
    exp_max = expected_max_t(n)     # 噪声地板(参考)
    bonf = bonferroni_t(n)          # 正式临界值
    hlz = 3.0                       # 经验门槛
    strict = max(bonf, hlz)         # 最终判定门槛
    passed = [r for r in trials if _t(r) > strict]

    if verbose:
        print(f"\n{'=' * 74}")
        print(f"因子试验登记册  —  累计 {n} 次试验 ({n_unique} 个不同因子)")
        print(f"{'=' * 74}")
        print(f"{'因子名':<16}{'持有期':>6}{'IC均值':>10}{'t统计量':>10}{'ICIR年化':>10}{'多空夏普':>10}")
        print("-" * 74)
        for r in sorted(trials, key=lambda x: -_t(x)):
            print(f"{r['因子名']:<16}{r['持有期']:>6}{r['IC均值']:>10}{r['t统计量']:>10}"
                  f"{r['ICIR年化']:>10}{r['多空夏普'] or '-':>10}")
        print("-" * 74)
        print(f"多重检验校正 (试验次数 N={n}):")
        print(f"  · 噪声地板(期望最大t)  {exp_max:>5.2f}   ← 纯噪声试{n}次,典型最好成绩就有这么高")
        print(f"  · 常规单次检验         {1.96:>5.2f}   ← 试了{n}次后此标准已失效")
        print(f"  · Harvey-Liu-Zhu经验值 {hlz:>5.2f}")
        print(f"  · Bonferroni(α=0.05)  {bonf:>5.2f}   ← 正式临界值")
        print(f"\n判定门槛 t > {strict:.2f}   通过的因子: "
              f"{', '.join(r['因子名'] for r in passed) if passed else '无'}")
        if not passed:
            print("  → 目前没有因子能在多重检验校正后站得住。这不是坏事，说明你没在自欺。")
        print("\n⚠提醒: 登记册只算被正式评估的配置。调参数/换股票池/换持有期的尝试也算试验，")
        print("  真实N往往比这个数大。少记 = 高估显著性。")

    return {"n_trials": n, "n_unique": n_unique, "expected_max_t": exp_max,
            "bonferroni_t": bonf, "threshold": strict, "passed": passed}


if __name__ == "__main__":
    report()
