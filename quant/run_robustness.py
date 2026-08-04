# -*- coding: utf-8 -*-
"""稳健性检验：中性化 + 成本 —— 因子能不能用的最后两道关

顺序很重要：
  1. 先看原始因子(毛收益) → 有没有信号
  2. 中性化后还在不在 → 是不是行业/规模的伪装
  3. 扣成本后还剩多少 → 能不能真金白银赚到
  任何一关掉光，这个因子就该扔掉。
"""
from __future__ import annotations

import os

from costs import cost_report
from data import build_panel, to_wide, liquidity_filter
from evaluate import evaluate_factor
from factors import reversal_20d
from neutralize import build_sector_map, build_size_factor, neutralize

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main(horizon: int = 20):
    panel = build_panel()
    prices = to_wide(panel, "adjclose")
    mask = liquidity_filter(panel, min_turnover_yen=1e8)
    sector = build_sector_map(level="sector17")
    size = build_size_factor(prices)

    raw = reversal_20d(prices)
    variants = [
        ("20日反转-原始", raw),
        ("20日反转-行业中性", neutralize(raw, sector=sector)),
        ("20日反转-行业+规模中性", neutralize(raw, sector=sector, size=size)),
    ]

    for name, f in variants:
        res = evaluate_factor(f, prices, name=name, horizon=horizon, rebalance=horizon,
                              mask=mask, universe="Prime", note="稳健性检验",
                              plot_path=os.path.join(OUT_DIR, f"{name}.png"))
        # 成本分析(用对齐后的因子)
        from evaluate import align, forward_returns
        fa, _ = align(f, forward_returns(prices, horizon), mask)
        rep = cost_report(res["quantile_returns"], fa, rebalance=horizon,
                          spread_bp=5.0, commission_bp=0.0)
        print(f"\n  ── 成本分析 ──")
        for k, v in rep.items():
            if isinstance(v, float):
                fmt = f"{v:+.2%}" if "年化" in k and "次数" not in k and "换手" not in k else f"{v:.2f}"
                print(f"  {k:<14}: {fmt}")


if __name__ == "__main__":
    main()
