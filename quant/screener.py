# -*- coding: utf-8 -*-
"""每日领域扫描：各领域筛出得分8分以上的股票

⚠必读：这个"得分"就是诊断页那10项判据的合计。它已经过全市场98万样本验证，
  得分高的股票后20日胜率52.9% < 随机买入54.6% (t=-3.88)——**没有预测力甚至反向**。
  所以本模块的正确定位是「今天哪些股票技术面处于强势状态」的**状态筛选器**，
  不是选股建议。同一份榜单里我一并给出行业超额收益，那个数是客观的相对强弱。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_validate_diagnosis import trend_score
from sectors_custom import ORDER, build_sectors


def scan(prices: pd.DataFrame, volume: pd.DataFrame, min_score: int = 8,
         top_n: int = 3, market: str = "prime") -> dict[str, list[dict]]:
    """扫描各领域，返回 {领域: [ {ticker,score,chg20,chg60,excess20,price}, ... ]}
    排序：得分降序 → 行业超额收益降序(同分时选相对强的)"""
    sectors = build_sectors(market)
    score = trend_score(prices, volume)
    last = score.index[-1]
    px = prices

    out = {}
    for name in ORDER:
        members = [t for t in sectors.get(name, []) if t in px.columns]
        if not members:
            out[name] = []
            continue
        # 该领域当日中位涨幅(算超额用)
        chg20_all = (px[members].iloc[-1] / px[members].iloc[-21] - 1)
        med20 = float(chg20_all.median())

        rows = []
        for t in members:
            s = score.at[last, t] if t in score.columns else np.nan
            if not np.isfinite(s) or s < min_score:
                continue
            c20 = float(px[t].iloc[-1] / px[t].iloc[-21] - 1)
            c60 = float(px[t].iloc[-1] / px[t].iloc[-61] - 1)
            rows.append({
                "ticker": t, "score": int(s), "price": float(px[t].iloc[-1]),
                "chg20": c20, "chg60": c60, "excess20": c20 - med20,
            })
        rows.sort(key=lambda r: (-r["score"], -r["excess20"]))
        out[name] = rows[:top_n]
    return out


def scan_summary(result: dict[str, list[dict]]) -> str:
    total = sum(len(v) for v in result.values())
    empty = [k for k, v in result.items() if not v]
    s = f"共 {total} 只入选"
    if empty:
        s += f"；无入选领域: {', '.join(empty)}"
    return s
