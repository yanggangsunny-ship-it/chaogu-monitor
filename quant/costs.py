# -*- coding: utf-8 -*-
"""交易成本模型：换手率 → 成本 → 净收益 → 盈亏平衡成本

为什么必须做：
  毛收益漂亮的因子，扣成本后经常一文不值。20日调仓的反转因子换手率极高，
  而它的多空毛年化只有+2.7% —— 成本能不能覆盖，是"能不能用"的分水岭。

日股成本构成(2026年现状)：
  1. 手续费：SBI/乐天等主要券商国内现货已**零手续费**(2023起)，信用交易同样。
     但机构/大额或其他券商仍有费用 → COMMISSION_BP 可配置，默认0。
  2. 买卖价差：东证tick规则按价位分档，流动股约1个tick。
     实务上单边成本≈半个价差 → 用 SPREAD_BP 近似(默认5bp=0.05%)。
  3. 冲击成本：你的委托吃掉多少流动性。平方根模型
     impact = k × σ × sqrt(参与率)，小资金可忽略，大资金是主导项。
  4. 信用交易另有利息(年率~2.8%)和贷株料，做空还有逆日步 → 多空组合要额外计。

⚠这个模型给的是**量级**不是精确值。真实成交价滑点受下单方式影响极大。
  更重要的用法是反过来问：**盈亏平衡成本是多少bp**？如果只有3bp，
  说明这个因子毫无空间，任何执行摩擦都会杀死它。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 245


def portfolio_turnover(factor: pd.DataFrame, n_groups: int = 5, rebalance: int = 20,
                       min_stocks: int = 30) -> pd.Series:
    """每个调仓日的单边换手率(0~1)：新持仓中有多少比例是新买入的。
    等权组合下 turnover = |新权重-旧权重|之和 / 2"""
    prev = {}
    rows = {}
    for i, dt in enumerate(factor.index):
        if i % rebalance:
            continue
        f = factor.loc[dt].dropna()
        if len(f) < min_stocks:
            continue
        q = pd.qcut(f.rank(method="first"), n_groups, labels=False)
        cur = {g: set(f.index[q == g]) for g in range(n_groups)}
        if prev:
            tos = []
            for g in cur:
                old, new = prev.get(g, set()), cur[g]
                if not new:
                    continue
                # 等权：旧持仓每只权重1/|old|，新持仓1/|new|
                changed = len(new - old) / len(new)
                tos.append(changed)
            if tos:
                rows[dt] = float(np.mean(tos))
        prev = cur
    return pd.Series(rows).sort_index()


def apply_costs(qr: pd.DataFrame, turnover: pd.Series, n_groups: int = 5,
                spread_bp: float = 5.0, commission_bp: float = 0.0,
                impact_bp: float = 0.0, margin_rate_annual: float = 0.028,
                rebalance: int = 20) -> pd.DataFrame:
    """把成本扣到分组收益上。
    单边成本(bp) = 价差/2 + 手续费 + 冲击；调仓日买卖各一次 → 双边
    多空组合额外计信用/融券利息(按日计提)"""
    one_way_bp = spread_bp / 2 + commission_bp + impact_bp
    net = qr.copy()
    cost_series = pd.Series(0.0, index=qr.index)
    for dt, to in turnover.items():
        if dt in cost_series.index:
            # 换手率 × 双边成本
            cost_series[dt] = to * 2 * one_way_bp / 1e4
    for col in net.columns:
        if col.startswith("Q") and "-" not in col:
            net[col] = net[col] - cost_series
    ls_col = f"Q{n_groups}-Q1"
    if ls_col in net.columns:
        # 多空：两腿都有成本 + 融资融券利息(按日)
        daily_rate = margin_rate_annual / TRADING_DAYS
        net[ls_col] = qr[ls_col] - cost_series * 2 - daily_rate
    return net


def breakeven_cost_bp(qr: pd.DataFrame, turnover: pd.Series, n_groups: int = 5,
                      rebalance: int = 20) -> float:
    """盈亏平衡的单边成本(bp)：成本超过此值，多空组合年化收益转负"""
    ls_col = f"Q{n_groups}-Q1"
    if ls_col not in qr.columns or qr[ls_col].empty:
        return float("nan")
    gross_daily = qr[ls_col].mean()
    if gross_daily <= 0:
        return 0.0
    n_rebal = max(len(turnover), 1)
    avg_to = turnover.mean() if len(turnover) else 0
    if avg_to == 0:
        return float("inf")
    # 年化毛收益 = 年化成本 时的单边bp
    rebalances_per_year = TRADING_DAYS / rebalance
    gross_annual = gross_daily * TRADING_DAYS
    return gross_annual / (avg_to * 2 * 2 * rebalances_per_year) * 1e4


def cost_report(qr: pd.DataFrame, factor: pd.DataFrame, n_groups: int = 5,
                rebalance: int = 20, spread_bp: float = 5.0,
                commission_bp: float = 0.0, margin_rate_annual: float = 0.028) -> dict:
    """成本分析总报告"""
    to = portfolio_turnover(factor, n_groups, rebalance)
    net = apply_costs(qr, to, n_groups, spread_bp, commission_bp,
                      margin_rate_annual=margin_rate_annual, rebalance=rebalance)
    ls = f"Q{n_groups}-Q1"
    out = {
        "平均单边换手率": to.mean() if len(to) else np.nan,
        "年化调仓次数": TRADING_DAYS / rebalance,
        "年化单边换手": to.mean() * TRADING_DAYS / rebalance if len(to) else np.nan,
        "盈亏平衡成本bp": breakeven_cost_bp(qr, to, n_groups, rebalance),
    }
    if ls in qr.columns and qr[ls].std():
        g, n = qr[ls], net[ls]
        out["多空毛年化"] = (1 + g.mean()) ** TRADING_DAYS - 1
        out["多空净年化"] = (1 + n.mean()) ** TRADING_DAYS - 1
        out["毛夏普"] = g.mean() / g.std() * np.sqrt(TRADING_DAYS)
        out["净夏普"] = n.mean() / n.std() * np.sqrt(TRADING_DAYS) if n.std() else np.nan
    return out
