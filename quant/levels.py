# -*- coding: utf-8 -*-
"""关键价位：支撑/压力 + 参考买卖位

方法(全部可复核，不用玄学):
  1. 摆动高低点(swing pivot)：局部极值，被反复测试的价位才算有效
  2. 成交量分布(volume profile)：把成交量按价格分箱，堆积最多的价位=筹码密集区，
     天然形成支撑/压力(套牢盘和获利盘都在那里)
  3. 均线：MA20/MA60是动态支撑压力
  4. 整数关口 + 52周高低

每个价位都给出**被测试次数**和**守住次数**——你能自己判断这条线可不可信，
而不是看我画一条线就信。

⚠免责：支撑压力是市场参与者行为的经验总结，不是物理定律。本模块只做客观计算，
  "买入位/卖出位"是**基于波动率的机械算法**，不构成任何建议。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> float:
    """平均真实波幅——决定止损该放多远的客观依据"""
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def swing_points(high: pd.Series, low: pd.Series, left: int = 5, right: int = 5) -> tuple:
    """摆动高低点：左右各left/right根K线内的局部极值"""
    hi, lo = [], []
    h, l = high.values, low.values
    for i in range(left, len(h) - right):
        w_h = h[i - left:i + right + 1]
        w_l = l[i - left:i + right + 1]
        if h[i] == w_h.max() and (w_h == h[i]).sum() == 1:
            hi.append((high.index[i], float(h[i])))
        if l[i] == w_l.min() and (w_l == l[i]).sum() == 1:
            lo.append((low.index[i], float(l[i])))
    return hi, lo


def _cluster(levels: list[tuple], tol_pct: float = 0.02) -> list[dict]:
    """把相近价位合并成一条线(容差2%)，记录被触及次数"""
    if not levels:
        return []
    vals = sorted(levels, key=lambda x: x[1])
    out, cur = [], [vals[0]]
    for d, v in vals[1:]:
        if abs(v - cur[-1][1]) / cur[-1][1] <= tol_pct:
            cur.append((d, v))
        else:
            out.append(cur)
            cur = [(d, v)]
    out.append(cur)
    res = []
    for g in out:
        dates = [d for d, _ in g if d is not None]   # 成交量节点无日期,需过滤
        res.append({"price": float(np.mean([v for _, v in g])), "touches": len(g),
                    "last_date": max(dates) if dates else None})
    return res


def volume_profile(close: pd.Series, volume: pd.Series, bins: int = 40,
                   lookback: int = 250) -> pd.DataFrame:
    """成交量分布：近lookback日按价格分箱累计成交量。
    量堆积最大的价位=筹码密集区(POC)，是最强的支撑/压力"""
    c, v = close.tail(lookback), volume.tail(lookback)
    ok = c.notna() & v.notna()
    c, v = c[ok], v[ok]
    if len(c) < 20:
        return pd.DataFrame(columns=["price", "volume"])
    edges = np.linspace(c.min(), c.max(), bins + 1)
    idx = np.clip(np.digitize(c.values, edges) - 1, 0, bins - 1)
    agg = np.zeros(bins)
    np.add.at(agg, idx, v.values)
    mid = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"price": mid, "volume": agg})


def key_levels(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
               lookback: int = 250, max_each: int = 4) -> dict:
    """汇总关键价位。返回 support/resistance 列表(按强度排序)+ 现价 + POC"""
    h, l, c = high.tail(lookback), low.tail(lookback), close.tail(lookback)
    px = float(c.iloc[-1])
    hi, lo = swing_points(h, l)
    clusters = _cluster(hi + lo)

    vp = volume_profile(close, volume, lookback=lookback)
    poc = float(vp.loc[vp["volume"].idxmax(), "price"]) if len(vp) else np.nan
    # 高量价位(前25%)也作为价位候选
    if len(vp):
        hot = vp[vp["volume"] >= vp["volume"].quantile(0.85)]["price"].tolist()
        clusters += [{"price": p, "touches": 0, "last_date": None, "vol_node": True}
                     for p in hot]
        clusters = _cluster([(d.get("last_date"), d["price"]) for d in clusters
                             if d["price"] == d["price"]])

    # 用历史检验每条线：价格接近后是否守住
    def _test(level: float, is_support: bool) -> tuple[int, int]:
        near = (c - level).abs() / level < 0.015
        tested = held = 0
        idxs = np.where(near.values)[0]
        for i in idxs:
            if i + 5 >= len(c):
                continue
            tested += 1
            fut = c.iloc[i + 1:i + 6]
            held += int(fut.min() > level * 0.98) if is_support else int(fut.max() < level * 1.02)
        return tested, held

    sup, res = [], []
    for cl in clusters:
        p = cl["price"]
        if not np.isfinite(p) or p <= 0:
            continue
        is_sup = p < px
        t, hd = _test(p, is_sup)
        item = {"price": p, "touches": cl["touches"], "tested": t, "held": hd,
                "dist_pct": (p - px) / px * 100,
                "is_poc": abs(p - poc) / poc < 0.02 if poc == poc else False}
        (sup if is_sup else res).append(item)

    sup.sort(key=lambda x: -x["price"])   # 最近的支撑在前
    res.sort(key=lambda x: x["price"])
    return {"price": px, "poc": poc, "support": sup[:max_each],
            "resistance": res[:max_each], "volume_profile": vp}


def trade_levels(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                 atr_mult_stop: float = 2.0, lookback: int = 250) -> dict:
    """机械算出的参考买卖位(不是建议):
      · 买入参考 = 最近支撑上方一点(回踩不破才进)
      · 止损     = 支撑下方 atr_mult_stop 倍ATR (避开日常波动)
      · 目标     = 最近压力位
      · 盈亏比   = (目标-买入)/(买入-止损)"""
    kl = key_levels(high, low, close, volume, lookback)
    px, a = kl["price"], atr(high, low, close)
    sup = kl["support"][0]["price"] if kl["support"] else px - 2 * a
    res = kl["resistance"][0]["price"] if kl["resistance"] else px + 2 * a
    entry = sup * 1.005
    stop = sup - atr_mult_stop * a
    rr = (res - entry) / (entry - stop) if entry > stop else np.nan
    return {**kl, "atr": a, "entry": entry, "stop": stop, "target": res,
            "risk_reward": rr,
            "stop_pct": (stop - entry) / entry * 100,
            "target_pct": (res - entry) / entry * 100}
