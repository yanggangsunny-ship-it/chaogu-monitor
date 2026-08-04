# -*- coding: utf-8 -*-
"""日频面板数据：date × ticker 的 OHLCV + 复权价，带本地parquet缓存

复权处理：用Yahoo的adjclose(已含分红+拆股调整)算收益率；OHLC原值另存(画图/看真实价位用)。
因子和收益率一律基于adjclose，避免除权日产生假跌幅。

停牌/缺失：某日无成交则该ticker当日为NaN，**不做前向填充**——填充会制造虚假的
"零收益日"污染因子和IC。需要连续序列的地方由使用方自己决定怎么处理。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

from universe import CACHE_DIR, load_universe

JST = timezone(timedelta(hours=9))
_PANEL_CACHE = os.path.join(CACHE_DIR, "panel_{market}_{years}y.parquet")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _fetch_one(session: requests.Session, ticker: str, years: int, retries: int = 2):
    """拉单只股票日线。返回DataFrame(date,ticker,open,high,low,close,adjclose,volume)或None"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": f"{years}y"}
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:  # 限流则退避重试
                time.sleep(2 + attempt * 3)
                continue
            res = r.json().get("chart", {}).get("result")
            if not res:
                return None
            d = res[0]
            ts = d.get("timestamp")
            if not ts:
                return None
            q = d["indicators"]["quote"][0]
            adj_block = d["indicators"].get("adjclose")
            adj = adj_block[0]["adjclose"] if adj_block else q["close"]
            df = pd.DataFrame(
                {
                    "date": [datetime.fromtimestamp(t, JST).date() for t in ts],
                    "ticker": ticker,
                    "open": q["open"],
                    "high": q["high"],
                    "low": q["low"],
                    "close": q["close"],
                    "adjclose": adj,
                    "volume": q["volume"],
                }
            )
            return df.dropna(subset=["close", "adjclose"])
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1 + attempt * 2)
    return None


def build_panel(market: str = "prime", years: int = 10, workers: int = 8,
                refresh: bool = False, limit: int | None = None) -> pd.DataFrame:
    """构建面板并缓存。返回长表(date,ticker,ohlcv,adjclose)"""
    cache = _PANEL_CACHE.format(market=market, years=years)
    if not refresh and os.path.exists(cache):
        return pd.read_parquet(cache)

    uni = load_universe((market,))
    tickers = uni["ticker"].tolist()
    if limit:
        tickers = tickers[:limit]

    session = requests.Session()
    session.headers.update(_UA)
    t0 = time.time()
    frames, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tk, df in zip(tickers, ex.map(lambda t: _fetch_one(session, t, years), tickers)):
            (frames.append(df) if df is not None and len(df) else failed.append(tk))

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    if not limit:
        panel.to_parquet(cache, index=False)
    print(f"[面板] {len(tickers)}只 → 成功{len(frames)} 失败{len(failed)} "
          f"| {panel['date'].min().date()}~{panel['date'].max().date()} "
          f"| {len(panel):,}行 | {time.time() - t0:.0f}秒")
    if failed:
        print(f"[面板] 失败样例: {failed[:8]}")
    return panel


# 单日涨跌超过此倍数视为数据错误(日股有涨跌停限制,正常单日不可能翻倍以上)
MAX_DAILY_MOVE = 1.0   # ±100%


def _scrub_bad_prices(px: pd.DataFrame) -> pd.DataFrame:
    """剔除数据源的坏价格。实测Yahoo复权价偶有天文数字(如8303.T出现539亿円/4.6e33,
    真实价约2730)，一个点就能污染整个统计。判据：单日涨跌>±100%且次日跌回 → 置NaN"""
    r = px.pct_change()
    bad = r.abs() > MAX_DAILY_MOVE
    # 反向跳变(次日又跳回)也标记，覆盖"错一天"和"错一段"两种情况
    bad = bad | bad.shift(-1).fillna(False)
    n = int(bad.sum().sum())
    if n:
        cols = px.columns[bad.any()].tolist()
        print(f"[数据清洗] 剔除 {n} 个异常价格点，涉及 {len(cols)} 只: {cols[:5]}")
    return px.mask(bad)


def to_wide(panel: pd.DataFrame, field: str = "adjclose", scrub: bool = True) -> pd.DataFrame:
    """长表→宽表 (index=date, columns=ticker)。价格字段默认做坏点清洗"""
    w = panel.pivot(index="date", columns="ticker", values=field).sort_index()
    if scrub and field in ("adjclose", "close", "open", "high", "low"):
        w = _scrub_bad_prices(w)
    return w


def daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """基于复权价的日收益率宽表。停牌日为NaN(不填充)"""
    px = to_wide(panel, "adjclose")
    return px.pct_change()


def liquidity_filter(panel: pd.DataFrame, min_turnover_yen: float = 1e8,
                     window: int = 20) -> pd.DataFrame:
    """流动性过滤掩码(True=可交易)：过去window日均成交额≥阈值。
    因子研究必须做——否则结果被无法实际成交的微型股主导"""
    turnover = to_wide(panel, "close") * to_wide(panel, "volume")
    avg = turnover.rolling(window, min_periods=max(5, window // 2)).mean()
    return avg >= min_turnover_yen


if __name__ == "__main__":
    p = build_panel(limit=30)
    px = to_wide(p)
    print(f"\n宽表: {px.shape[0]}个交易日 × {px.shape[1]}只")
    print(f"缺失率: {px.isna().mean().mean():.2%}")
    print(px.iloc[-3:, :5].round(1).to_string())
