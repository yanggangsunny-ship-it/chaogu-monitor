# -*- coding: utf-8 -*-
"""把统计术语翻译成人话 —— 外行一眼能懂，不用查资料

核心思路：t统计量本质是回答"这个规律有多大可能只是碰巧"。
把它换算成「纯属巧合的概率」和「相当于抛硬币多少次的偏差」，就直观了。
"""
from __future__ import annotations

import math
from statistics import NormalDist


def p_value(t: float) -> float:
    """双尾p值 = 如果其实没规律(纯随机)，却观察到这么强结果的概率"""
    return 2 * (1 - NormalDist().cdf(abs(t)))


def odds_text(p: float) -> str:
    """概率 → 「约1/N」这种直观说法"""
    if p <= 0 or p != p:
        return "小于千万分之一"
    n = 1 / p
    if n >= 1e7:
        return "小于千万分之一"
    if n >= 1e4:
        return f"约 1/{n / 1e4:.0f}万"
    if n >= 1000:
        return f"约 1/{n / 1000:.1f}千"
    return f"约 1/{n:.0f}"


def confidence_label(t: float) -> str:
    """|t| → 可信度档位(外行版)"""
    a = abs(t)
    if a < 1.5:
        return "看不出规律"
    if a < 2.0:
        return "有微弱迹象，多半是运气"
    if a < 3.0:
        return "有迹象，但试多了就不算数"
    if a < 4.0:
        return "相当可靠"
    return "几乎可以确定"


def explain_t(t: float, subject: str = "这个指标", positive_means: str = "越高越涨",
              negative_means: str = "越高反而越跌") -> str:
    """一句话人话解释。例:
    explain_t(-5.87,'同行超额') →
    '同行超额: 越高反而越跌 — 几乎可以确定(纯属巧合的概率小于千万分之一)'"""
    if abs(t) < 1.5:
        return f"{subject}: 看不出规律，和随机猜没区别"
    direction = positive_means if t > 0 else negative_means
    return (f"{subject}: {direction} — {confidence_label(t)}"
            f"(纯属巧合的概率{odds_text(p_value(t))})")


def coin_flip_analogy(t: float, n_trials: int = 1000) -> str:
    """换算成抛硬币的直觉：n次里正面多出多少个才有同等异常度"""
    excess = t * math.sqrt(n_trials) / 2
    return (f"相当于抛{n_trials}次硬币，正面比反面多出约{abs(excess):.0f}个"
            f"—— 这种偏差{'很难' if abs(t) > 3 else '有可能'}用运气解释")


def verdict_line(t: float, n_trials_done: int = 1) -> str:
    """结合已做试验次数给判定(多重检验)"""
    from research_log import bonferroni_t
    thr = max(bonferroni_t(n_trials_done), 3.0)
    if abs(t) > thr:
        return f"✓ 通过检验(试了{n_trials_done}次，门槛|t|>{thr:.1f}，本次{abs(t):.2f})"
    return (f"✗ 没通过(试了{n_trials_done}次，门槛|t|>{thr:.1f}，本次仅{abs(t):.2f})"
            f" — 试的次数越多，门槛越高，否则总能碰巧撞出一个好看的结果")


if __name__ == "__main__":
    for t, name in [(-5.87, "同行超额高"), (-3.88, "10项判据得分"), (5.22, "20日反转"),
                    (1.96, "60日反转"), (-1.32, "12-1动量")]:
        print(f"t={t:>6.2f}  {explain_t(t, name)}")
    print()
    print(coin_flip_analogy(-5.87))
    print(coin_flip_analogy(1.96))
