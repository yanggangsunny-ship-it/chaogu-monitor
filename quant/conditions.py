# -*- coding: utf-8 -*-
"""选股条件库：可自由组合的原子条件

每个条件 = (字段, 比较符, 阈值)。字段全部只用 ≤t 日的信息(无前视)。
组合方式 = 全部满足(AND)。这是最透明的组合方式——你能说清楚为什么选中某只股。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 字段定义: 名称 -> (计算函数, 单位, 说明)
# 计算函数签名 f(px, vol, sector) -> 宽表(date × ticker)


def _rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _sector_excess(px: pd.DataFrame, sector: pd.Series, window: int) -> pd.DataFrame:
    chg = px / px.shift(window) - 1
    cols = chg.columns.intersection(sector.dropna().index)
    grp = sector.reindex(cols)
    med = chg[cols].T.groupby(grp).median().T          # date × 行业
    bench = pd.DataFrame({t: med[grp[t]] for t in cols if grp[t] in med.columns})
    return chg[bench.columns] - bench


FIELDS = {
    "5日涨幅%": (lambda px, v, s: (px / px.shift(5) - 1) * 100, "%", "短期动能"),
    "20日涨幅%": (lambda px, v, s: (px / px.shift(20) - 1) * 100, "%", "月度涨跌，反转因子的核心"),
    "60日涨幅%": (lambda px, v, s: (px / px.shift(60) - 1) * 100, "%", "季度趋势"),
    "250日涨幅%": (lambda px, v, s: (px / px.shift(250) - 1) * 100, "%", "年度趋势"),
    "RSI14": (lambda px, v, s: _rsi(px), "", "超买超卖，<30超卖 >70超买"),
    "距MA20%": (lambda px, v, s: (px / px.rolling(20, min_periods=10).mean() - 1) * 100,
                "%", "现价偏离20日均线"),
    "距MA60%": (lambda px, v, s: (px / px.rolling(60, min_periods=30).mean() - 1) * 100,
                "%", "现价偏离60日均线"),
    "距52周高%": (lambda px, v, s: (px / px.rolling(250, min_periods=100).max() - 1) * 100,
                  "%", "离一年最高还差多少(负值)"),
    "距52周低%": (lambda px, v, s: (px / px.rolling(250, min_periods=100).min() - 1) * 100,
                  "%", "比一年最低高多少"),
    "量比": (lambda px, v, s: v / v.rolling(20, min_periods=10).mean(), "倍", "当日量/20日均量"),
    "20日波动率%": (lambda px, v, s: px.pct_change().rolling(20, min_periods=10).std() * 100,
                    "%", "日均波动幅度，越大越颠"),
    "同行超额20日%": (lambda px, v, s: _sector_excess(px, s, 20) * 100, "%", "相对同行业中位数"),
    "同行超额60日%": (lambda px, v, s: _sector_excess(px, s, 60) * 100, "%", "相对同行业中位数"),
    "日均成交额(亿円)": (lambda px, v, s: (px * v).rolling(20, min_periods=10).mean() / 1e8,
                         "亿", "流动性，太小的买不进去"),
}

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b}


def compute_field(name: str, px: pd.DataFrame, vol: pd.DataFrame,
                  sector: pd.Series | None) -> pd.DataFrame:
    fn = FIELDS[name][0]
    return fn(px, vol, sector)


def build_mask(conditions: list[dict], px: pd.DataFrame, vol: pd.DataFrame,
               sector: pd.Series | None, cache: dict | None = None) -> pd.DataFrame:
    """conditions=[{field, op, value}] → 全部满足的布尔宽表"""
    cache = cache if cache is not None else {}
    out = None
    for c in conditions:
        f = cache.get(c["field"])
        if f is None:
            f = compute_field(c["field"], px, vol, sector)
            cache[c["field"]] = f
        m = OPS[c["op"]](f, c["value"])
        m = m.reindex_like(px).fillna(False)
        out = m if out is None else (out & m)
    if out is None:
        out = pd.DataFrame(True, index=px.index, columns=px.columns)
    return out & px.notna()
