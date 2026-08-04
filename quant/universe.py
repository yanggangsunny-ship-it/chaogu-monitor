# -*- coding: utf-8 -*-
"""股票池：从JPX官方上市银柄一览构建universe(带业种/规模分类)

⚠生存者偏差(survivorship bias)警告：
JPX这份名录是**当前时点快照**，只含现在还在上市的公司。过去10年退市/破产/被收购的公司
不在其中。用它回测会系统性高估收益(退市股通常先暴跌)。这是免费数据源的硬伤，
要根治需要 point-in-time 数据库(J-Quants付费档/Refinitiv等)。
本模块把universe抽象出来，将来换数据源只需替换 load_universe()。
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

JPX_MASTER_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
_MASTER_CACHE = os.path.join(CACHE_DIR, "jpx_master.parquet")

# 市场区分(只保留内国普通股，排除ETF/REIT/外国株/PRO Market)
MARKETS = {
    "prime": "プライム（内国株式）",
    "standard": "スタンダード（内国株式）",
    "growth": "グロース（内国株式）",
}


def load_master(refresh: bool = False) -> pd.DataFrame:
    """JPX上市银柄一览。列: code/name/market/sector33/sector17/size"""
    if not refresh and os.path.exists(_MASTER_CACHE):
        return pd.read_parquet(_MASTER_CACHE)

    resp = requests.get(JPX_MASTER_URL, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    raw = pd.read_excel(io.BytesIO(resp.content), dtype={"コード": str})
    df = raw.rename(
        columns={
            "日付": "asof",
            "コード": "code",
            "銘柄名": "name",
            "市場・商品区分": "market",
            "33業種区分": "sector33",
            "17業種区分": "sector17",
            "規模区分": "size",
        }
    )[["asof", "code", "name", "market", "sector33", "sector17", "size"]]
    df["code"] = df["code"].astype(str).str.strip()
    df.to_parquet(_MASTER_CACHE, index=False)
    return df


def load_universe(markets=("prime",), refresh: bool = False) -> pd.DataFrame:
    """指定市场的内国普通股列表。ticker列=Yahoo格式(如 6954.T)"""
    df = load_master(refresh=refresh)
    wanted = [MARKETS[m] for m in markets]
    out = df[df["market"].isin(wanted)].copy()
    out["ticker"] = out["code"] + ".T"
    return out.reset_index(drop=True)


if __name__ == "__main__":
    u = load_universe(("prime",))
    print(f"Prime内国株: {len(u)}只  (asof {u['asof'].iloc[0]})")
    print(u[["ticker", "name", "sector17", "size"]].head(8).to_string(index=False))
    print("\n规模分布:", u["size"].value_counts().to_dict())
