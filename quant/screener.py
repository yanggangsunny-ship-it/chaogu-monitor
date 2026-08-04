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


# 两种筛选方向(2026-08-04实测,Prime全市场)
#   强势模式: 选同行超额高的。⚠历史检验 t=-5.87 显著为负 —— 近期跑赢同行的股票
#            后20日倾向回吐，这是历史上**吃亏**的一边。仅适合"我就想跟强势"的场景。
#   反转模式: 选同行超额低的(跌得比同行多)。方向与验证一致(20日反转 t=+5.22)。
#            但注意:该信号做系统化高换手策略时会被成本吃光(净夏普-0.12);
#            人工少量持有、低换手的场景成本低得多，不能直接照搬那个结论。
MODE_STRONG = "强势(超额高)"
MODE_REVERSAL = "反转(超额低)"


def entry_zone(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
               tol: float = 0.05) -> dict | None:
    """算该股的买入参考位，并判断现价是否落在其±tol范围内。
    买入参考=最近支撑上方0.5%(见levels.trade_levels)"""
    try:
        from levels import trade_levels
        r = trade_levels(high.dropna(), low.dropna(), close.dropna(), volume.dropna())
    except Exception:
        return None
    entry, px = r["entry"], r["price"]
    if not (entry and entry == entry and px):
        return None
    gap = (px - entry) / entry          # 现价相对买入位的偏离
    return {"entry": entry, "gap": gap, "in_zone": abs(gap) <= tol,
            "stop": r["stop"], "target": r["target"], "rr": r["risk_reward"]}


def scan(prices: pd.DataFrame, volume: pd.DataFrame, min_score: int = 8,
         top_n: int = 3, market: str = "prime", mode: str = MODE_STRONG,
         min_excess: float = 0.0, highs: pd.DataFrame | None = None,
         lows: pd.DataFrame | None = None, entry_tol: float = 0.05) -> dict[str, list[dict]]:
    """扫描各领域，返回 {领域: [ {ticker,score,chg20,chg60,excess20,price}, ... ]}
    mode=强势 → 只留 超额≥min_excess，按超额降序
    mode=反转 → 只留 超额≤-min_excess，按超额升序(跌得越多越前)"""
    sectors = build_sectors(market)
    score = trend_score(prices, volume)
    last = score.index[-1]
    px = prices
    reversal = mode == MODE_REVERSAL

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
            if not np.isfinite(c20):
                continue
            exc = c20 - med20
            if reversal:
                if exc > -min_excess:      # 反转模式:要跑输同行到一定程度
                    continue
            elif exc < min_excess:         # 强势模式:要跑赢同行到一定程度
                continue
            c60 = float(px[t].iloc[-1] / px[t].iloc[-61] - 1)
            rows.append({
                "ticker": t, "score": int(s), "price": float(px[t].iloc[-1]),
                "chg20": c20, "chg60": c60, "excess20": exc,
            })
        rows.sort(key=lambda r: (-r["score"], r["excess20"] if reversal else -r["excess20"]))
        rows = rows[:top_n]
        # 只对入选的少数股票算买入区(trade_levels较重,不能全池跑)
        if highs is not None and lows is not None:
            for r in rows:
                t = r["ticker"]
                if t in highs.columns and t in lows.columns:
                    ez = entry_zone(highs[t], lows[t], px[t], volume[t], entry_tol)
                    if ez:
                        r.update({"entry": ez["entry"], "entry_gap": ez["gap"],
                                  "in_zone": ez["in_zone"], "stop": ez["stop"],
                                  "target": ez["target"], "rr": ez["rr"]})
        out[name] = rows
    return out


def scan_summary(result: dict[str, list[dict]]) -> str:
    total = sum(len(v) for v in result.values())
    empty = [k for k, v in result.items() if not v]
    s = f"共 {total} 只入选"
    if empty:
        s += f"；无入选领域: {', '.join(empty)}"
    return s
