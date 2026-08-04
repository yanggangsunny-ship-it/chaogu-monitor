# -*- coding: utf-8 -*-
"""个股趋势诊断：用一张清单回答「这只股票是不是进入上升趋势」

设计原则：不给玄学结论，每条判断都摆出具体数字，你自己能复核。
最后再用**历史统计**回答关键问题：这只股票以前出现同样形态时，后面涨的概率多大。
那个数字才是量化能提供、看图看不出来的东西。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_checks(prices: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """逐日算出10条判据的布尔矩阵(全部只用当日及之前数据)"""
    ma5, ma20, ma60 = _sma(prices, 5), _sma(prices, 20), _sma(prices, 60)
    vol20 = volume.rolling(20, min_periods=10).mean()
    rsi = _rsi(prices)
    chk = pd.DataFrame(index=prices.index)
    chk["站上20日线"] = prices > ma20
    chk["站上60日线"] = prices > ma60
    chk["20日线向上"] = ma20 > ma20.shift(5)
    chk["60日线向上"] = ma60 > ma60.shift(10)
    chk["多头排列"] = (ma5 > ma20) & (ma20 > ma60)
    chk["20日涨幅为正"] = prices > prices.shift(20)
    chk["60日涨幅为正"] = prices > prices.shift(60)
    chk["量能配合"] = volume > vol20            # 当日量高于20日均量
    chk["RSI>50"] = rsi > 50
    chk["未跌破近20日低点"] = prices > prices.rolling(20, min_periods=10).min() * 1.02
    return chk


def verdict_from_score(score: int, total: int = 10) -> tuple[str, str]:
    """得分 → 结论 + 说明"""
    if score >= 8:
        return "上升趋势确立", "多项指标同向向上，趋势明确"
    if score >= 6:
        return "偏强/趋势形成中", "多数指标转好，但尚未全面确认"
    if score >= 4:
        return "横盘整理", "多空指标交织，方向未明"
    if score >= 2:
        return "偏弱", "多数指标向下，反弹需确认"
    return "下降趋势", "指标全面向下，不宜逆势"


def diagnose(prices: pd.Series, volume: pd.Series, horizon: int = 20,
             strong_score: int = 8) -> dict:
    """完整诊断。返回结论/得分/清单明细/历史同形态统计"""
    prices, volume = prices.dropna(), volume.reindex(prices.index)
    if len(prices) < 80:
        return {"error": "数据不足(需至少80个交易日)"}

    chk = build_checks(prices, volume)
    today = chk.index[-1]
    row = chk.loc[today]
    score = int(row.sum())
    verdict, note = verdict_from_score(score, len(row))

    # 明细：每条判据附具体数值
    ma5, ma20, ma60 = _sma(prices, 5), _sma(prices, 20), _sma(prices, 60)
    px = prices.iloc[-1]
    vol20 = volume.rolling(20, min_periods=10).mean()
    rsi = _rsi(prices)
    detail = {
        "站上20日线": f"现价{px:,.0f} vs MA20 {ma20.iloc[-1]:,.0f} ({px / ma20.iloc[-1] - 1:+.1%})",
        "站上60日线": f"现价{px:,.0f} vs MA60 {ma60.iloc[-1]:,.0f} ({px / ma60.iloc[-1] - 1:+.1%})",
        "20日线向上": f"MA20 五日变化 {ma20.iloc[-1] / ma20.iloc[-6] - 1:+.2%}",
        "60日线向上": f"MA60 十日变化 {ma60.iloc[-1] / ma60.iloc[-11] - 1:+.2%}",
        "多头排列": f"MA5 {ma5.iloc[-1]:,.0f} / MA20 {ma20.iloc[-1]:,.0f} / MA60 {ma60.iloc[-1]:,.0f}",
        "20日涨幅为正": f"20日涨幅 {px / prices.iloc[-21] - 1:+.1%}",
        "60日涨幅为正": f"60日涨幅 {px / prices.iloc[-61] - 1:+.1%}",
        "量能配合": f"当日量 {volume.iloc[-1]:,.0f} = 20日均量的 {volume.iloc[-1] / vol20.iloc[-1]:.1f}倍",
        "RSI>50": f"RSI(14) = {rsi.iloc[-1]:.0f}",
        "未跌破近20日低点": f"近20日最低 {prices.tail(20).min():,.0f}，现价高出 {px / prices.tail(20).min() - 1:+.1%}",
    }

    # 历史同形态统计：过去出现同等得分时，之后horizon日的表现
    scores = chk.sum(axis=1)
    fwd = prices.shift(-horizon) / prices - 1
    hist = fwd[(scores >= strong_score) & fwd.notna()]
    base = fwd.dropna()
    stats = {
        "样本数": len(hist),
        "胜率": float((hist > 0).mean()) if len(hist) else np.nan,
        "平均收益": float(hist.mean()) if len(hist) else np.nan,
        "中位数收益": float(hist.median()) if len(hist) else np.nan,
        "基准胜率": float((base > 0).mean()) if len(base) else np.nan,
        "基准平均": float(base.mean()) if len(base) else np.nan,
    }
    return {"verdict": verdict, "note": note, "score": score, "total": len(row),
            "checks": row.to_dict(), "detail": detail, "hist": stats,
            "date": today, "price": float(px), "series": prices,
            "ma": (ma5, ma20, ma60), "scores": scores, "horizon": horizon,
            "strong_score": strong_score}
