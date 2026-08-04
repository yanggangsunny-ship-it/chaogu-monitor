# -*- coding: utf-8 -*-
"""个股信号点位：把因子在某只股票上的触发时点标注到价格图上

用途：回测统计量(IC/夏普)是全市场平均的抽象结果，看不出"在这只股票上到底
什么时候该买"。本模块把抽象因子落到具体点位，用于直觉校验和复盘。

信号定义：t日该股因子值在全市场截面中排进前 top_pct(默认20%) → 买入信号
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_points(factor: pd.DataFrame, ticker: str, top_pct: float = 0.2,
                  mask: pd.DataFrame | None = None, min_gap: int = 5) -> pd.DataFrame:
    """返回该股票的信号点 DataFrame(date, rank_pct, factor)。
    min_gap=信号最小间隔交易日数(避免连续多日重复标注)"""
    if ticker not in factor.columns:
        return pd.DataFrame(columns=["date", "rank_pct", "factor"])
    f = factor if mask is None else factor.where(mask.reindex_like(factor).fillna(False))
    # 逐日截面百分位排名(1=最高)
    pct = f.rank(axis=1, pct=True, ascending=False)
    s_pct, s_val = pct[ticker], f[ticker]
    hit = s_pct <= top_pct
    rows, last_i = [], -10 ** 9
    idx = list(f.index)
    for i, dt in enumerate(idx):
        if bool(hit.get(dt, False)) and (i - last_i) >= min_gap:
            rows.append({"date": dt, "rank_pct": float(s_pct[dt]), "factor": float(s_val[dt])})
            last_i = i
    return pd.DataFrame(rows)


def forward_outcome(prices: pd.DataFrame, ticker: str, points: pd.DataFrame,
                    horizon: int = 20) -> pd.DataFrame:
    """每个信号点之后 horizon 日的实际收益 → 直观检验信号成色"""
    if points.empty or ticker not in prices.columns:
        return points
    px = prices[ticker]
    fwd = px.shift(-horizon) / px - 1
    out = points.copy()
    out[f"{horizon}日后收益"] = [float(fwd.get(d, np.nan)) for d in out["date"]]
    return out
