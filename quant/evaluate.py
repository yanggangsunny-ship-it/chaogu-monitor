# -*- coding: utf-8 -*-
"""因子评估：IC / ICIR / 分组收益 / 衰减曲线 —— 全部手写，不依赖alphalens

方法论要点(每一条都是常见踩坑点)：
1. 前视偏差：t日因子只能预测 t→t+h 的收益。fwd_ret 用 shift(-h) 对齐，且
   t日因子必须只用 ≤t 的信息(由因子构造方保证)。
2. IC 用 Spearman 秩相关：对异常值稳健，且我们关心的是排序能力而非线性关系。
3. ICIR = mean(IC)/std(IC)，年化乘 sqrt(每年期数)。这是信号稳定性的核心指标
   (Grinold-Kahn: IR ≈ IC × sqrt(breadth))。
4. t统计量 = mean(IC)/std(IC)*sqrt(N)，用于判断 IC 是否显著非零。
   ⚠但单因子 t>2 不代表可用：如果你试了50个因子挑出这一个，多重检验下
   门槛要提高(López de Prado: 试N次后的 deflated Sharpe)。记录你试过多少个。
5. 分组收益等权、每期再平衡，Q5-Q1多空组合是因子纯度的直观体现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

TRADING_DAYS = 245  # 日股年均交易日


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """t日的前瞻收益 = t→t+horizon 的复权价收益率(宽表)"""
    return prices.shift(-horizon) / prices - 1


def align(factor: pd.DataFrame | pd.Series, fwd: pd.DataFrame,
          mask: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把因子和前瞻收益对齐到同一 (date × ticker) 网格；mask=可交易掩码"""
    if isinstance(factor, pd.Series):  # MultiIndex(date,ticker) → 宽表
        factor = factor.unstack()
    idx = factor.index.intersection(fwd.index)
    cols = factor.columns.intersection(fwd.columns)
    f, r = factor.loc[idx, cols], fwd.loc[idx, cols]
    if mask is not None:
        m = mask.reindex(index=idx, columns=cols).fillna(False)
        f, r = f.where(m), r.where(m)
    valid = f.notna() & r.notna()
    return f.where(valid), r.where(valid)


def compute_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_stocks: int = 30) -> pd.Series:
    """逐日截面 Spearman 秩相关 → IC 时序"""
    ics = {}
    fr, rr = factor.rank(axis=1), fwd.rank(axis=1)   # 秩变换后算Pearson=Spearman
    for dt in fr.index:
        a, b = fr.loc[dt], rr.loc[dt]
        ok = a.notna() & b.notna()
        if ok.sum() < min_stocks:
            continue
        x, y = a[ok], b[ok]
        sx, sy = x.std(), y.std()
        if sx == 0 or sy == 0:
            continue
        ics[dt] = ((x - x.mean()) * (y - y.mean())).mean() / (sx * sy) * len(x) / (len(x) - 1)
    return pd.Series(ics).sort_index()


def ic_summary(ic: pd.Series, horizon: int = 1) -> dict:
    """IC统计量。periods_per_year 按调仓周期折算"""
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    ppy = TRADING_DAYS / horizon
    return {
        "IC均值": mean,
        "IC标准差": std,
        "ICIR(单期)": mean / std if std else np.nan,
        "ICIR(年化)": mean / std * np.sqrt(ppy) if std else np.nan,
        "t统计量": mean / std * np.sqrt(n) if std else np.nan,
        "IC>0占比": (ic > 0).mean(),
        "样本期数": n,
    }


def quantile_returns(factor: pd.DataFrame, ret_1d: pd.DataFrame, n_groups: int = 5,
                     rebalance: int = 1, min_stocks: int = 30) -> pd.DataFrame:
    """分组组合日收益。每 rebalance 日按因子分n组(等权)，持有至下次调仓。
    返回 DataFrame(index=date, columns=[Q1..Qn, Q5-Q1])"""
    dates = factor.index
    holdings = {}          # 当前各组持仓
    rows = {}
    for i, dt in enumerate(dates):
        # ⚠顺序关键：先用「上一期形成的持仓」结算今日收益，再更新持仓。
        # 因子用t日收盘价算出，只能从 t+1 日开始赚钱；若先更新再结算会引入1日前视偏差
        # (反转因子受害最严重：当日下跌→因子值高→却被计入当日跌幅，导致IC与分组结论相反)
        if holdings and dt in ret_1d.index:
            r = ret_1d.loc[dt]
            row = {}
            for g, members in holdings.items():
                vals = r.reindex(members).dropna()
                row[f"Q{g + 1}"] = vals.mean() if len(vals) else np.nan
            if row:
                rows[dt] = row
        if i % rebalance == 0:                       # 调仓日：重新分组(从下一日开始生效)
            f = factor.loc[dt].dropna()
            if len(f) >= min_stocks:
                q = pd.qcut(f.rank(method="first"), n_groups, labels=False)
                holdings = {g: f.index[q == g].tolist() for g in range(n_groups)}
    df = pd.DataFrame(rows).T.sort_index()
    if not df.empty and f"Q{n_groups}" in df and "Q1" in df:
        df[f"Q{n_groups}-Q1"] = df[f"Q{n_groups}"] - df["Q1"]
    return df


def ic_decay(factor: pd.DataFrame, prices: pd.DataFrame, horizons=(1, 5, 10, 20, 40, 60),
             mask: pd.DataFrame | None = None) -> pd.Series:
    """不同持有期的IC均值 → 信号衰减速度"""
    out = {}
    for h in horizons:
        f, r = align(factor, forward_returns(prices, h), mask)
        ic = compute_ic(f, r)
        out[h] = ic.mean() if len(ic) else np.nan
    return pd.Series(out, name="IC均值")


def evaluate_factor(factor, prices: pd.DataFrame, name: str = "factor",
                    horizon: int = 1, n_groups: int = 5, rebalance: int | None = None,
                    mask: pd.DataFrame | None = None, plot_path: str | None = None,
                    universe: str = "", note: str = "", log: bool = True) -> dict:
    """一站式评估：打印统计量 + 画三联图(IC时序/分组净值/衰减)。返回结果dict。
    log=True 时自动登记到 factor_trials.csv (多重检验校正需要准确的试验次数)"""
    rebalance = rebalance or horizon
    fwd = forward_returns(prices, horizon)
    f, r = align(factor, fwd, mask)

    ic = compute_ic(f, r)
    stats = ic_summary(ic, horizon)
    ret_1d = prices.pct_change()
    qr = quantile_returns(f, ret_1d, n_groups, rebalance)
    decay = ic_decay(f, prices, mask=mask)

    print(f"\n{'=' * 62}\n因子: {name}   (持有期{horizon}日, 调仓{rebalance}日, {n_groups}分组)\n{'=' * 62}")
    for k, v in stats.items():
        print(f"  {k:<12}: {v:>8.4f}" if isinstance(v, float) else f"  {k:<12}: {v:>8}")
    ls_ann = ls_sharpe = float("nan")
    if not qr.empty:
        ann = (1 + qr.mean()) ** TRADING_DAYS - 1
        print(f"\n  分组年化收益: " + " | ".join(f"{c}:{ann[c]:+.1%}" for c in qr.columns))
        ls = qr.get(f"Q{n_groups}-Q1")
        if ls is not None and ls.std():
            ls_ann = (1 + ls.mean()) ** TRADING_DAYS - 1
            ls_sharpe = ls.mean() / ls.std() * np.sqrt(TRADING_DAYS)
            print(f"  多空组合 年化{ls_ann:+.1%} 夏普{ls_sharpe:.2f} "
                  f"最大回撤{((1 + ls).cumprod() / (1 + ls).cumprod().cummax() - 1).min():.1%}")
    print(f"\n  IC衰减: " + " | ".join(f"{h}日:{v:+.4f}" for h, v in decay.items()))

    if log:  # 自动登记，多重检验校正依赖准确的试验次数
        from research_log import log_trial, bonferroni_t, expected_max_t, load_trials
        period = f"{f.index.min().date()}~{f.index.max().date()}" if len(f.index) else ""
        log_trial(name, stats, universe=universe, horizon=horizon, rebalance=rebalance,
                  period=period, ls_ann=ls_ann, ls_sharpe=ls_sharpe, note=note)
        n = len(load_trials())
        thr = max(bonferroni_t(n), 3.0)
        verdict = "✓ 站得住" if abs(stats["t统计量"]) > thr else "✗ 校正后不显著"
        print(f"\n  [登记册] 第 {n} 次试验 | 噪声地板t={expected_max_t(n):.2f} "
              f"| 判定门槛t>{thr:.2f} | 本因子 {verdict}")

    if plot_path:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
        ax = axes[0]
        ic.plot(ax=ax, lw=0.7, alpha=0.55, label="IC")
        ic.rolling(60, min_periods=20).mean().plot(ax=ax, lw=1.8, color="#e03131", label="60期均值")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name} — IC时序 (均值{stats['IC均值']:.4f}, t={stats['t统计量']:.1f})")
        ax.legend(fontsize=8)
        ax = axes[1]
        for c in qr.columns:
            if c.startswith("Q") and "-" not in c:
                (1 + qr[c]).cumprod().plot(ax=ax, lw=1.2, label=c)
        ax.set_title("分组累计净值(等权)")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax = axes[2]
        decay.plot(ax=ax, marker="o")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("IC衰减 (持有期→IC均值)")
        ax.set_xlabel("持有期(日)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  图已保存: {plot_path}")

    return {"ic": ic, "stats": stats, "quantile_returns": qr, "decay": decay}
