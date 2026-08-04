# -*- coding: utf-8 -*-
"""选股观察记录：把「我挑一只看看」变成能积累证据的实验

纪律(照做才有意义)：
  1. **记录必须在事前**——挑中当天就登记，价格自动锁定，不能事后补记或修改
  2. **不能挑挑拣拣**——涨的记下来、跌的删掉，等于自欺；本模块不提供"改结果"功能
  3. **对照基准**——每条记录同时算日经和同行业的同期涨跌，只有跑赢基准才算本事
  4. **样本量**——单只30天说明不了任何事(基准胜率54.6%)。20~30条之后才有参考价值
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


TRACK_PATH = os.path.join(_base_dir(), "tracker.json")
HORIZONS = (5, 10, 20, 30)      # 观察节点(交易日)


def load() -> list[dict]:
    try:
        with open(TRACK_PATH, encoding="utf-8") as f:
            return json.load(f).get("picks", [])
    except (OSError, json.JSONDecodeError):
        return []


def save(picks: list[dict]) -> None:
    with open(TRACK_PATH, "w", encoding="utf-8") as f:
        json.dump({"picks": picks}, f, ensure_ascii=False, indent=1)


def add_pick(ticker: str, name: str, price: float, date: str, **meta) -> dict:
    """登记一条观察(事前记录,价格锁定)"""
    picks = load()
    rec = {
        "id": f"{ticker}_{date}",
        "date": date, "ticker": ticker, "name": name,
        "price": float(price),
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    rec.update({k: v for k, v in meta.items() if v is not None})
    if any(p["id"] == rec["id"] for p in picks):
        return rec        # 同股同日不重复登记
    picks.append(rec)
    save(picks)
    return rec


def remove(pick_id: str) -> None:
    save([p for p in load() if p.get("id") != pick_id])


def evaluate(picks: list[dict], prices, sector_map=None, bench: str = "^N225") -> list[dict]:
    """算每条记录的持有收益 + 基准对比。prices=宽表(date×ticker)"""
    import numpy as np
    import pandas as pd

    out = []
    idx = prices.index
    for p in picks:
        r = dict(p)
        tk = p["ticker"]
        d0 = pd.Timestamp(p["date"])
        if tk not in prices.columns or d0 > idx.max():
            r["error"] = "无数据"
            out.append(r)
            continue
        pos = idx.searchsorted(d0)
        s = prices[tk]
        p0 = float(p["price"])
        cur_i = len(idx) - 1
        r["days"] = int(cur_i - pos)                    # 已过交易日
        cur = float(s.iloc[cur_i]) if pd.notna(s.iloc[cur_i]) else np.nan
        r["current"] = cur
        r["ret"] = (cur / p0 - 1) if cur == cur else np.nan

        # 基准同期
        for label, col in [("bench", bench)]:
            if col in prices.columns:
                b = prices[col]
                b0, b1 = b.iloc[pos], b.iloc[cur_i]
                r[f"{label}_ret"] = (b1 / b0 - 1) if pd.notna(b0) and pd.notna(b1) else np.nan
        # 行业同期中位
        if sector_map is not None and tk in sector_map.index:
            peers = [t for t in sector_map[sector_map == sector_map[tk]].index
                     if t in prices.columns]
            if len(peers) >= 3:
                sub = prices[peers]
                med = (sub.iloc[cur_i] / sub.iloc[pos] - 1).median()
                r["sector_ret"] = float(med)
        if r.get("ret") == r.get("ret") and r.get("bench_ret") == r.get("bench_ret"):
            r["alpha"] = r["ret"] - r["bench_ret"]

        # 各节点收益(到期才有值)
        for h in HORIZONS:
            j = pos + h
            r[f"d{h}"] = float(s.iloc[j] / p0 - 1) if j <= cur_i and pd.notna(s.iloc[j]) else np.nan
        out.append(r)
    return out


def summary(rows: list[dict]) -> dict:
    """战绩汇总——样本少时不要当真"""
    import numpy as np

    done = [r for r in rows if r.get("ret") == r.get("ret")]
    if not done:
        return {"n": 0}
    rets = np.array([r["ret"] for r in done])
    alphas = np.array([r["alpha"] for r in done if r.get("alpha") == r.get("alpha")])
    mature = [r for r in done if r.get("d30") == r.get("d30")]
    return {
        "n": len(done),
        "胜率": float((rets > 0).mean()),
        "平均收益": float(rets.mean()),
        "中位收益": float(np.median(rets)),
        "跑赢基准率": float((alphas > 0).mean()) if len(alphas) else np.nan,
        "平均超额": float(alphas.mean()) if len(alphas) else np.nan,
        "满30日数": len(mature),
        "满30日胜率": float(np.mean([r["d30"] > 0 for r in mature])) if mature else np.nan,
        "满30日均收益": float(np.mean([r["d30"] for r in mature])) if mature else np.nan,
    }
