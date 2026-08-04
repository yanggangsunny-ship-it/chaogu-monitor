# -*- coding: utf-8 -*-
"""上涨预期分：按实测方向给权重带符号的预测型评分(0~100)

与「技术面强势分」的区别：
  强势分  = 描述现状(涨得好=高分)，实测方向为负 → 高分反而后续跌
  上涨预期分 = 直接以「预测后续上涨」为目标，权重带符号：
             实测正向的判据加分，负向的判据反过来加分(即不满足时加分)

关键纪律 —— 必须样本外验证：
  权重在**训练期**测出来，效果在**测试期**检验。
  若只在同一段数据上又算权重又看效果，那是拟合历史，不是发现规律。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate import align, compute_ic, forward_returns, ic_summary


def build_raw_criteria(px: pd.DataFrame, vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """原始判据(布尔)。方向由权重符号决定，这里不预设看涨看跌"""
    ma5 = px.rolling(5, min_periods=3).mean()
    ma20 = px.rolling(20, min_periods=10).mean()
    ma60 = px.rolling(60, min_periods=30).mean()
    vol20 = vol.rolling(20, min_periods=10).mean()
    d = px.diff()
    up_d = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn_d = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up_d / dn_d.replace(0, np.nan))
    low20 = px.rolling(20, min_periods=10).min()
    high20 = px.rolling(20, min_periods=10).max()
    up_day = px > px.shift(1)
    big_vol = vol > vol20

    return {
        "站上20日线": px > ma20,
        "站上60日线": px > ma60,
        "20日线向上": ma20 > ma20.shift(5),
        "60日线向上": ma60 > ma60.shift(10),
        "多头排列": (ma5 > ma20) & (ma20 > ma60),
        "20日涨幅为正": px > px.shift(20),
        "60日涨幅为正": px > px.shift(60),
        "放量上涨": big_vol & up_day,
        "放量下跌": big_vol & (~up_day),
        "RSI>50": rsi > 50,
        "RSI超卖(<30)": rsi < 30,
        "接近20日低点": px < low20 * 1.03,
        "接近20日高点": px > high20 * 0.97,
    }


def measure_weights(px: pd.DataFrame, vol: pd.DataFrame, mask: pd.DataFrame,
                    horizon: int, start=None, end=None) -> pd.DataFrame:
    """在指定区间测每条判据的t统计量 → 带符号权重(合计绝对值100)"""
    fwd = forward_returns(px, horizon)
    sl = slice(start, end)
    rows = []
    for name, chk in build_raw_criteria(px, vol).items():
        f = chk.astype(float).where(px.notna())
        fa, ra = align(f.loc[sl], fwd.loc[sl], mask.loc[sl])
        ic = compute_ic(fa, ra)
        if not len(ic):
            continue
        st = ic_summary(ic, horizon)
        rows.append({"判据": name, "t": st["t统计量"], "IC": st["IC均值"]})
    df = pd.DataFrame(rows)
    df["强度"] = df["t"].abs()
    df["权重"] = (df["强度"] / df["强度"].sum() * 100 * np.sign(df["t"])).round(1)
    return df.sort_values("强度", ascending=False)


def compute_score(px: pd.DataFrame, vol: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """按带符号权重算分，再线性映射到0~100(便于阅读)"""
    crit = build_raw_criteria(px, vol)
    w = dict(zip(weights["判据"], weights["权重"]))
    raw = sum(crit[n].astype(float) * wt for n, wt in w.items() if n in crit)
    lo = sum(min(wt, 0) for wt in w.values())      # 理论最低分
    hi = sum(max(wt, 0) for wt in w.values())      # 理论最高分
    valid = px.notna() & px.rolling(60, min_periods=30).mean().notna()
    return ((raw - lo) / (hi - lo) * 100).where(valid)
