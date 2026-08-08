# -*- coding: utf-8 -*-
"""涨跌分档 → 后续表现统计(可自选回看/持有天数)

回答的问题：过去N天涨/跌了多少的股票，之后M天通常怎么走。
分档统计比单一相关系数直观——能看出「涨>50%」那种胜率低但均值高的彩票型分布。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BINS = [-1, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 10]
LABELS = ["跌>20%", "跌10-20%", "跌0-10%", "涨0-10%",
          "涨10-20%", "涨20-30%", "涨30-50%", "涨>50%"]


def compute(prices: pd.DataFrame, mask: pd.DataFrame | None = None,
            lookback: int = 20, horizon: int = 20) -> tuple[pd.DataFrame, dict]:
    """返回(分档统计表, 各档收益序列dict用于画分布图)"""
    chg = (prices / prices.shift(lookback) - 1)
    fwd = prices.shift(-horizon) / prices - 1
    if mask is not None:
        chg, fwd = chg.where(mask), fwd.where(mask)
    df = pd.DataFrame({"chg": chg.stack(), "fwd": fwd.stack()}).dropna()
    if df.empty:
        return pd.DataFrame(), {}
    df["b"] = pd.cut(df["chg"], BINS, labels=LABELS)

    rows, series = [], {}
    for b, g in df.groupby("b", observed=True):
        f = g["fwd"]
        up, dn = f[f > 0], f[f <= 0]
        rows.append({
            "过去N天": b, "样本数": len(g),
            "上涨概率": (f > 0).mean(), "平均收益": f.mean(), "中位数": f.median(),
            "赢时均涨": up.mean() if len(up) else np.nan,
            "输时均跌": dn.mean() if len(dn) else np.nan,
            "盈亏比": (up.mean() / abs(dn.mean())) if len(up) and len(dn) and dn.mean() else np.nan,
            "最差5%": f.quantile(0.05), "最好5%": f.quantile(0.95),
        })
        series[str(b)] = f.values
    tbl = pd.DataFrame(rows)
    all_f = df["fwd"]
    tbl.loc[len(tbl)] = {
        "过去N天": "— 全样本 —", "样本数": len(all_f),
        "上涨概率": (all_f > 0).mean(), "平均收益": all_f.mean(),
        "中位数": all_f.median(), "赢时均涨": np.nan, "输时均跌": np.nan,
        "盈亏比": np.nan, "最差5%": np.nan, "最好5%": np.nan,
    }
    series["— 全样本 —"] = all_f.values
    return tbl, series
