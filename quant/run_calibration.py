# -*- coding: utf-8 -*-
"""校准实验：在日股Prime全市场复现12-1动量与20日反转。

这一步的目的不是"发现赚钱因子"，而是**验证数据管道和评估代码是否正确**。
教科书结论(美股/多数发达市场)：
  - 12-1动量: 月频IC为正, 长期Q5>Q1
  - 短期反转: 20日尺度IC为正(跌多反弹)
  ⚠日本是著名的"动量失效市场"(Asness et al.对日股动量弱有专门讨论)，
   若动量偏弱/为负，未必是代码错——但反转应当显著为正。
"""
from __future__ import annotations

import os

from data import build_panel, to_wide, liquidity_filter
from evaluate import evaluate_factor
from factors import momentum_12_1, reversal_20d

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def main(market: str = "prime", years: int = 10, horizon: int = 20):
    panel = build_panel(market=market, years=years)
    prices = to_wide(panel, "adjclose")
    mask = liquidity_filter(panel, min_turnover_yen=1e8)   # 日均成交额≥1亿円
    print(f"股票池: {prices.shape[1]}只 × {prices.shape[0]}交易日 "
          f"| 流动性过滤后日均可交易 {mask.sum(axis=1).mean():.0f}只")

    for name, factor in [
        ("12-1月动量", momentum_12_1(prices)),
        ("20日反转", reversal_20d(prices)),
    ]:
        evaluate_factor(
            factor, prices, name=name, horizon=horizon, rebalance=horizon,
            mask=mask, plot_path=os.path.join(OUT_DIR, f"{name}.png"),
        )


if __name__ == "__main__":
    main()
