# -*- coding: utf-8 -*-
"""测量10项判据各自的预测力 → 据此分配100分权重

方法：把每项判据单独当成0/1因子跑IC，看它自己有多少信息含量。
权重不按主观重要性给，按**实测的信息含量**给——这样才有依据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import build_panel, liquidity_filter, to_wide
from evaluate import align, compute_ic, forward_returns, ic_summary


def build_criteria(px: pd.DataFrame, vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """10项判据的布尔矩阵(与diagnosis.py/screener.py口径一致)"""
    ma5 = px.rolling(5, min_periods=3).mean()
    ma20 = px.rolling(20, min_periods=10).mean()
    ma60 = px.rolling(60, min_periods=30).mean()
    vol20 = vol.rolling(20, min_periods=10).mean()
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    low20 = px.rolling(20, min_periods=10).min()
    return {
        "站上20日线": px > ma20,
        "站上60日线": px > ma60,
        "20日线向上": ma20 > ma20.shift(5),
        "60日线向上": ma60 > ma60.shift(10),
        "多头排列": (ma5 > ma20) & (ma20 > ma60),
        "20日涨幅为正": px > px.shift(20),
        "60日涨幅为正": px > px.shift(60),
        "量能配合": vol > vol20,
        "RSI>50": rsi > 50,
        "未跌破近20日低点": px > low20 * 1.02,
    }


def main(horizon: int = 20):
    panel = build_panel()
    px = to_wide(panel, "adjclose")
    vol = to_wide(panel, "volume", scrub=False)
    mask = liquidity_filter(panel, min_turnover_yen=1e8)
    fwd = forward_returns(px, horizon)
    valid = px.notna() & px.rolling(60, min_periods=30).mean().notna()

    rows = []
    for name, chk in build_criteria(px, vol).items():
        f = chk.astype(float).where(valid)
        fa, ra = align(f, fwd, mask)
        ic = compute_ic(fa, ra)
        st = ic_summary(ic, horizon)
        # 满足/不满足两组的后续收益差(更直观)
        yes = ra.where(fa == 1).stack().dropna()
        no = ra.where(fa == 0).stack().dropna()
        rows.append({
            "判据": name, "IC均值": st["IC均值"], "t统计量": st["t统计量"],
            "满足组胜率": (yes > 0).mean(), "不满足组胜率": (no > 0).mean(),
            "胜率差": (yes > 0).mean() - (no > 0).mean(),
            "满足组均收益": yes.mean(), "不满足组均收益": no.mean(),
            "收益差": yes.mean() - no.mean(),
        })
    df = pd.DataFrame(rows)
    df["信息含量"] = df["t统计量"].abs()
    df["权重"] = (df["信息含量"] / df["信息含量"].sum() * 100).round(0).astype(int)
    # 修正凑整误差，保证合计100
    diff = 100 - df["权重"].sum()
    if diff:
        df.loc[df["信息含量"].idxmax(), "权重"] += diff
    df = df.sort_values("信息含量", ascending=False)

    print(f"\n{'判据':<16}{'t统计量':>9}{'IC均值':>9}{'满足组胜率':>10}{'不满足':>8}{'胜率差':>8}{'权重':>6}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r['判据']:<16}{r['t统计量']:>9.2f}{r['IC均值']:>9.4f}"
              f"{r['满足组胜率']:>10.1%}{r['不满足组胜率']:>8.1%}{r['胜率差']:>+8.1%}{r['权重']:>6}")
    print("-" * 70)
    print(f"{'合计':<16}{'':>26}{'':>18}{df['权重'].sum():>6}")
    print("\n注: t为负=该判据满足时后续反而跌(日股反转特性)。权重按|t|(信息含量)分配,")
    print("    与方向无关——权重高只代表这条判据「有话说」,不代表它说的是「会涨」。")
    df.to_csv("output/criteria_weights.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    main()
