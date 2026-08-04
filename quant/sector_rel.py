# -*- coding: utf-8 -*-
"""行业相对强度：把股票放回它自己的行业里比较

为什么这比全市场比较更有意义：
  三菱重工跌8%，如果整个防卫板块跌10%，它其实是**相对强势**；
  反之在半导体狂涨30%的月份，某半导体股只涨5%，绝对看是涨，相对看是掉队。
  行业相对强度剥离了板块β，剩下的才是这只股票自己的东西。

两个用法：
  1. RS(相对强度) = 个股收益 - 行业中位数收益 → 看它跑赢还是跑输同行
  2. 行业内排名 = 该股在本行业N只股票中的位次
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sector_median_returns(prices: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """各行业每日中位收益(宽表: date × 行业)"""
    ret = prices.pct_change()
    cols = ret.columns.intersection(sector.dropna().index)
    grp = sector.reindex(cols)
    return ret[cols].T.groupby(grp).median().T


def relative_strength(prices: pd.DataFrame, sector: pd.Series,
                      window: int = 60) -> pd.DataFrame:
    """滚动window日的超额收益(个股累计收益 - 所属行业中位数累计收益)"""
    ret = prices.pct_change()
    sec_ret = sector_median_returns(prices, sector)
    cols = ret.columns.intersection(sector.dropna().index)
    bench = pd.DataFrame({t: sec_ret[sector[t]] for t in cols if sector[t] in sec_ret.columns})
    excess = ret[bench.columns] - bench
    return excess.rolling(window, min_periods=window // 2).sum()


def sector_rank(values: pd.Series, sector: pd.Series) -> pd.Series:
    """截面上按行业分组做百分位排名(1=行业内最强)"""
    df = pd.DataFrame({"v": values, "s": sector.reindex(values.index)}).dropna()
    return df.groupby("s")["v"].rank(pct=True, ascending=False)


def peer_snapshot(prices: pd.DataFrame, sector: pd.Series, ticker: str,
                  windows=(20, 60, 250), top_n: int = 8) -> dict:
    """该股在本行业中的位置：各周期涨幅、行业内排名、同行对比表"""
    if ticker not in sector.index:
        return {"error": f"{ticker} 无行业分类"}
    sec = sector[ticker]
    peers = [t for t in sector[sector == sec].index if t in prices.columns]
    if len(peers) < 3:
        return {"error": f"行业「{sec}」样本不足({len(peers)}只)"}

    rows = {}
    for w in windows:
        chg = prices[peers].iloc[-1] / prices[peers].iloc[-(w + 1)] - 1
        rows[f"{w}日涨幅"] = chg
    tbl = pd.DataFrame(rows).dropna(how="all")
    ranks = {c: int(tbl[c].rank(ascending=False)[ticker]) if ticker in tbl.index and
             tbl[c].notna()[ticker] else None for c in tbl.columns}

    key = f"{windows[1]}日涨幅"
    lead = tbl.sort_values(key, ascending=False).head(top_n)
    return {"sector": sec, "n_peers": len(peers), "table": tbl, "ranks": ranks,
            "self": tbl.loc[ticker] if ticker in tbl.index else None,
            "sector_median": tbl.median(), "leaders": lead}
