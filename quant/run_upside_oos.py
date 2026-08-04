# -*- coding: utf-8 -*-
"""上涨预期分 —— 训练期定权重，测试期检验(样本外)

这是判断「这个分数到底有没有用」的唯一诚实方法。
在训练期算权重、又在训练期看效果，必然好看——那是拟合不是发现。
"""
from __future__ import annotations

import pandas as pd

from costs import cost_report
from data import build_panel, liquidity_filter, to_wide
from evaluate import (TRADING_DAYS, align, compute_ic, forward_returns,
                      ic_summary, quantile_returns)
from upside_score import compute_score, measure_weights

SPLIT = "2023-01-01"      # 训练: 2016-08~2022-12  测试: 2023-01~今


def main(horizon: int = 20):
    panel = build_panel()
    px = to_wide(panel, "adjclose")
    vol = to_wide(panel, "volume", scrub=False)
    mask = liquidity_filter(panel, min_turnover_yen=1e8)

    print(f"训练期: {px.index.min().date()} ~ {pd.Timestamp(SPLIT).date()}")
    print(f"测试期: {pd.Timestamp(SPLIT).date()} ~ {px.index.max().date()}\n")

    w = measure_weights(px, vol, mask, horizon, end=SPLIT)
    print("【训练期实测的判据权重】(正=该判据满足时后续涨, 负=满足时后续跌)")
    print(f"{'判据':<16}{'t统计量':>9}{'权重':>8}")
    print("-" * 34)
    for _, r in w.iterrows():
        print(f"{r['判据']:<16}{r['t']:>9.2f}{r['权重']:>8.1f}")

    score = compute_score(px, vol, w)
    fwd = forward_returns(px, horizon)

    for label, sl in [("训练期(拟合,必然好看)", slice(None, SPLIT)),
                      ("★测试期(样本外,真考验)", slice(SPLIT, None))]:
        fa, ra = align(score.loc[sl], fwd.loc[sl], mask.loc[sl])
        ic = compute_ic(fa, ra)
        st = ic_summary(ic, horizon)
        qr = quantile_returns(fa, px.loc[sl].pct_change(), 5, horizon)
        ann = (1 + qr.mean()) ** TRADING_DAYS - 1 if not qr.empty else {}
        print(f"\n══ {label} ══")
        print(f"  IC均值 {st['IC均值']:+.4f}   t统计量 {st['t统计量']:+.2f}   "
              f"ICIR年化 {st['ICIR(年化)']:+.2f}")
        if len(ann):
            print("  分组年化: " + " | ".join(f"{c}:{ann[c]:+.1%}" for c in qr.columns))
        # 高分组 vs 全样本
        hi = ra.where(fa >= 70).stack().dropna()
        allp = ra.stack().dropna()
        if len(hi):
            print(f"  70分以上: {len(hi):,}个样本  胜率{(hi > 0).mean():.1%}  "
                  f"平均{hi.mean():+.2%}")
            print(f"  全样本  : {len(allp):,}个样本  胜率{(allp > 0).mean():.1%}  "
                  f"平均{allp.mean():+.2%}")
            print(f"  → 超额胜率 {(hi > 0).mean() - (allp > 0).mean():+.1%}   "
                  f"超额收益 {hi.mean() - allp.mean():+.2%}")
        if label.startswith("★") and not qr.empty:
            rep = cost_report(qr, fa, rebalance=horizon, spread_bp=5.0)
            print(f"  成本后: 多空毛年化{rep.get('多空毛年化', 0):+.2%} → "
                  f"净年化{rep.get('多空净年化', 0):+.2%}  "
                  f"(净夏普{rep.get('净夏普', float('nan')):.2f}, "
                  f"盈亏平衡成本{rep.get('盈亏平衡成本bp', 0):.1f}bp)")
    w.to_csv("output/upside_weights.csv", index=False, encoding="utf-8-sig")
    return w


if __name__ == "__main__":
    main()
