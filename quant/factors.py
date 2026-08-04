# -*- coding: utf-8 -*-
"""因子构造。所有因子在 t 日只使用 ≤t 的信息(防前视)"""
from __future__ import annotations

import pandas as pd


def momentum_12_1(prices: pd.DataFrame, lookback: int = 245, skip: int = 20) -> pd.DataFrame:
    """12-1月动量：过去12个月涨幅，跳过最近1个月(避开短期反转干扰)。
    t日值 = (P[t-skip] / P[t-lookback]) - 1 —— 全部用历史价，无前视"""
    return prices.shift(skip) / prices.shift(lookback) - 1


def reversal_20d(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """20日反转：过去20日涨幅取负号(跌得多的排前面 → 预期反弹)"""
    return -(prices / prices.shift(window) - 1)


def volatility(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """低波动因子：过去60日收益率标准差取负(低波动排前面)"""
    return -prices.pct_change().rolling(window, min_periods=window // 2).std()


def to_series(wide: pd.DataFrame, name: str = "factor") -> pd.Series:
    """宽表 → MultiIndex(date,ticker) Series"""
    return wide.stack().rename(name)
