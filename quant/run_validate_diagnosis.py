# -*- coding: utf-8 -*-
"""验证「趋势诊断10项判据」到底有没有预测力

把诊断得分(0-10)当成一个因子，扔进已建好的评估管道跑全市场。
这是对自己拍脑袋定的判据做的诚实检验——如果它在1559只股票上没有超额，
那"8项达标=上升趋势"这个说法就该扔掉或重做。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from costs import cost_report
from data import build_panel, liquidity_filter, to_wide
from evaluate import align, evaluate_factor, forward_returns

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def _rsi_wide(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def trend_score(px: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """向量化的10项判据得分(与diagnosis.py逐股版本口径一致)"""
    ma5 = px.rolling(5, min_periods=3).mean()
    ma20 = px.rolling(20, min_periods=10).mean()
    ma60 = px.rolling(60, min_periods=30).mean()
    vol20 = vol.rolling(20, min_periods=10).mean()
    rsi = _rsi_wide(px)
    low20 = px.rolling(20, min_periods=10).min()

    checks = [
        px > ma20,                       # 1 站上20日线
        px > ma60,                       # 2 站上60日线
        ma20 > ma20.shift(5),            # 3 20日线向上
        ma60 > ma60.shift(10),           # 4 60日线向上
        (ma5 > ma20) & (ma20 > ma60),    # 5 多头排列
        px > px.shift(20),               # 6 20日涨幅为正
        px > px.shift(60),               # 7 60日涨幅为正
        vol > vol20,                     # 8 量能配合
        rsi > 50,                        # 9 RSI>50
        px > low20 * 1.02,               # 10 未跌破近20日低点
    ]
    valid = px.notna() & ma60.notna()
    score = sum(c.astype(float) for c in checks)
    return score.where(valid)


def main(horizon: int = 20):
    panel = build_panel()
    px = to_wide(panel, "adjclose")
    vol = to_wide(panel, "volume")
    mask = liquidity_filter(panel, min_turnover_yen=1e8)

    score = trend_score(px, vol)
    print(f"得分分布(全样本): " + " ".join(
        f"{int(k)}分:{v / score.notna().sum().sum():.1%}"
        for k, v in score.stack().value_counts().sort_index().items()))

    res = evaluate_factor(score, px, name="趋势诊断得分(10项)", horizon=horizon,
                          rebalance=horizon, mask=mask, universe="Prime",
                          note="验证自定判据", plot_path=os.path.join(OUT_DIR, "趋势诊断得分.png"))

    fa, _ = align(score, forward_returns(px, horizon), mask)
    rep = cost_report(res["quantile_returns"], fa, rebalance=horizon, spread_bp=5.0)
    print("\n  ── 成本分析 ──")
    for k, v in rep.items():
        print(f"  {k:<14}: " + (f"{v:+.2%}" if ("年化" in k and "次数" not in k and "换手" not in k)
                                else f"{v:.2f}"))

    # 直接回答用户的问题：得分≥8 vs 全样本 的胜率对比
    fwd = forward_returns(px, horizon).where(mask)
    strong = fwd.where(fa >= 8).stack().dropna()
    allp = fwd.stack().dropna()
    print(f"\n  ── 「8项以上达标」全市场检验 ──")
    print(f"  强势样本  : {len(strong):,} 个  胜率 {(strong > 0).mean():.1%}  平均 {strong.mean():+.2%}")
    print(f"  全样本对照: {len(allp):,} 个  胜率 {(allp > 0).mean():.1%}  平均 {allp.mean():+.2%}")
    print(f"  → 超额胜率 {(strong > 0).mean() - (allp > 0).mean():+.1%}  "
          f"超额收益 {strong.mean() - allp.mean():+.2%}")


if __name__ == "__main__":
    main()
