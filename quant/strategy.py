# -*- coding: utf-8 -*-
"""策略回测：把「选中的股票等权持有N天」这件事诚实地算一遍

强制包含三道关(缺一道都会自欺)：
  1. 样本外  —— 训练期/测试期分开看，只在训练期好看的策略不算数
  2. 交易成本 —— 换手率×成本，很多策略毛收益漂亮净收益为负
  3. 基准对比 —— 跟"全市场等权"比，跑不赢基准的策略没有存在意义
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

TRADING_DAYS = 245


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _find(fname: str) -> str | None:
    d = _base_dir()
    for base in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    return None


STRAT_PATH = os.path.join(_base_dir(), "strategies.json")


def load_all() -> dict:
    p = _find("strategies.json")
    if p:
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_strategy(name: str, conditions: list[dict], settings: dict) -> None:
    d = load_all()
    d[name] = {"conditions": conditions, "settings": settings}
    with open(_find("strategies.json") or STRAT_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def delete_strategy(name: str) -> None:
    d = load_all()
    d.pop(name, None)
    with open(_find("strategies.json") or STRAT_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def backtest(sel: pd.DataFrame, px: pd.DataFrame, hold: int = 20,
             max_pos: int = 0, cost_bp: float = 10.0,
             start=None, end=None) -> dict:
    """sel=每日是否入选的布尔宽表。等权持有hold日后换仓。
    max_pos>0 时只取入选股中前max_pos只(按代码序,避免引入额外择股逻辑)"""
    sl = slice(start, end)
    s, p = sel.loc[sl], px.loc[sl]
    ret1 = p.pct_change()
    dates = p.index
    holdings, prev = [], set()
    rows, turns, counts = {}, [], []

    for i, dt in enumerate(dates):
        # 先用上期持仓结算今日收益(防1日前视)，再换仓
        if holdings:
            r = ret1.loc[dt].reindex(holdings).dropna()
            if len(r):
                rows[dt] = float(r.mean())
        if i % hold == 0:
            picks = s.loc[dt]
            names = list(picks[picks].index)
            if max_pos:
                names = names[:max_pos]
            counts.append(len(names))
            if names:
                cur = set(names)
                if prev:
                    turns.append(len(cur - prev) / len(cur))
                prev = cur
                holdings = names
            else:
                holdings = []      # 无股可选=空仓
                prev = set()

    if not rows:
        return {"error": "无有效持仓"}
    daily = pd.Series(rows).sort_index()
    bench = ret1.mean(axis=1).reindex(daily.index)      # 全市场等权基准

    turnover = float(np.mean(turns)) if turns else 0.0
    rebal_per_year = TRADING_DAYS / hold
    cost_annual = turnover * 2 * cost_bp / 1e4 * rebal_per_year
    ann = (1 + daily.mean()) ** TRADING_DAYS - 1
    bench_ann = (1 + bench.mean()) ** TRADING_DAYS - 1
    sharpe = daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() else np.nan
    cum = (1 + daily).cumprod()
    return {
        "daily": daily, "bench": bench,
        "年化收益": ann, "扣成本年化": ann - cost_annual,
        "基准年化": bench_ann, "超额年化": ann - bench_ann,
        "扣成本超额": ann - cost_annual - bench_ann,
        "夏普": sharpe, "最大回撤": float((cum / cum.cummax() - 1).min()),
        "胜率(日)": float((daily > 0).mean()),
        "平均持仓数": float(np.mean(counts)) if counts else 0,
        "换手率": turnover, "年化成本": cost_annual,
        "调仓次数": len(counts), "交易日数": len(daily),
    }
