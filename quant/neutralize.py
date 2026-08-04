# -*- coding: utf-8 -*-
"""因子中性化：剥离行业和规模暴露，看因子的"纯度"

为什么必须做：
  原始因子的收益可能只是在赌行业或赌小盘。比如"低波动因子"在日股常常等价于
  "买公用事业+食品"，你以为发现了波动率异象，其实只是行业配置。
  中性化后如果IC归零 → 这个因子没有独立信息，只是行业/规模的代理变量。

方法：逐日截面回归 factor ~ 行业哑变量 + log(市值)，取**残差**作为中性化因子。
  用最小二乘闭式解(手写,不引sklearn)，缺失值逐日剔除。

⚠市值数据的局限：Yahoo只给**当前**股本，历史市值用 现市值×(历史价/现价) 近似。
  回购/增发会让股本变化，越往前误差越大。size中性化因此是"近似中性"，
  用于判断因子是否被规模主导足够，不适合做精确的风险归因。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

from universe import CACHE_DIR, load_universe

_MKTCAP_CACHE = os.path.join(CACHE_DIR, "marketcap_snapshot.parquet")


def fetch_marketcap(market: str = "prime", refresh: bool = False) -> pd.DataFrame:
    """当前市值快照 (ticker, marketcap, price)。批量v7 quote，80只/批"""
    if not refresh and os.path.exists(_MKTCAP_CACHE):
        return pd.read_parquet(_MKTCAP_CACHE)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import market as mkt  # 复用已有的带crumb会话

    tickers = load_universe((market,))["ticker"].tolist()
    session, crumb = mkt._yahoo_session()
    rows = []
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        r = session.get("https://query1.finance.yahoo.com/v7/finance/quote",
                        params={"symbols": ",".join(chunk), "crumb": crumb}, timeout=40)
        for q in r.json().get("quoteResponse", {}).get("result", []):
            if q.get("marketCap") and q.get("regularMarketPrice"):
                rows.append({"ticker": q["symbol"], "marketcap": float(q["marketCap"]),
                             "price_now": float(q["regularMarketPrice"])})
    df = pd.DataFrame(rows)
    df.to_parquet(_MKTCAP_CACHE, index=False)
    return df


def build_size_factor(prices: pd.DataFrame, market: str = "prime") -> pd.DataFrame:
    """历史log市值(近似) = log(现市值 × 历史价/现价)，宽表"""
    mc = fetch_marketcap(market).set_index("ticker")
    cols = prices.columns.intersection(mc.index)
    scale = (mc.loc[cols, "marketcap"] / mc.loc[cols, "price_now"])   # ≈股本数
    cap = prices[cols].mul(scale, axis=1)
    return np.log(cap.where(cap > 0))   # 非正值(停牌/异常)置NaN,避免log告警


def build_sector_map(market: str = "prime", level: str = "sector17") -> pd.Series:
    """ticker → 行业分类"""
    u = load_universe((market,))
    return u.set_index("ticker")[level]


def _ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """最小二乘残差 y - X(X'X)^-1 X'y，用lstsq避免奇异矩阵问题"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def neutralize(factor: pd.DataFrame, sector: pd.Series | None = None,
               size: pd.DataFrame | None = None, min_stocks: int = 30) -> pd.DataFrame:
    """逐日截面中性化，返回残差因子(与输入同形状)。
    sector=ticker→行业 的Series; size=log市值宽表。两者可单独或同时使用"""
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    sec_dummies = None
    if sector is not None:
        aligned = sector.reindex(factor.columns)
        sec_dummies = pd.get_dummies(aligned, dummy_na=False).astype(float)  # ticker × 行业

    for dt in factor.index:
        f = factor.loc[dt].dropna()
        if len(f) < min_stocks:
            continue
        parts = [np.ones((len(f), 1))]                       # 截距
        if sec_dummies is not None:
            d = sec_dummies.reindex(f.index).fillna(0.0)
            if d.shape[1] > 1:
                parts.append(d.values[:, 1:])                # 去掉一列防共线
        if size is not None and dt in size.index:
            s = size.loc[dt].reindex(f.index)
            ok = s.notna()
            f, s = f[ok], s[ok]
            if len(f) < min_stocks:
                continue
            parts = [p[ok.values] if p.shape[0] == len(ok) else p for p in parts]
            parts[0] = np.ones((len(f), 1))
            if sec_dummies is not None and len(parts) > 1:
                parts[1] = sec_dummies.reindex(f.index).fillna(0.0).values[:, 1:]
            z = (s - s.mean()) / s.std() if s.std() else s * 0
            parts.append(z.values.reshape(-1, 1))
        X = np.hstack(parts)
        try:
            out.loc[dt, f.index] = _ols_residual(f.values.astype(float), X)
        except np.linalg.LinAlgError:
            continue
    return out
