# -*- coding: utf-8 -*-
"""持仓管理 + 基于T1/T2/T3的盈亏情景测算

与「买入参考位」的区别：
  买入参考是给「还没买」的股票算进场点；
  这里是给「已经持有」的股票算——成本已经固定，只关心从现价到各目标位/支撑位，
  账面盈亏会变成多少。两者的盈亏比含义不同，不要混用。
"""
from __future__ import annotations

import json
import os
import sys


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


PORTFOLIO_PATH = os.path.join(_base_dir(), "portfolio.json")


def _candidate_dirs() -> list[str]:
    """可能存放数据/源码的目录：exe所在目录、其上一级、再上一级。
    (打包后exe在 quant\\dist\\，market.py在 chaogu\\，差两级)"""
    d = _base_dir()
    return [d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))]


def _seed() -> list[dict]:
    """首次运行的种子：从 chaogu/market.py 的 POSITIONS 读现有持仓(只读)"""
    import ast
    for base in _candidate_dirs():
        path = os.path.join(base, "market.py")
        if not os.path.exists(path):
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                        getattr(t, "id", "") == "POSITIONS" for t in node.targets):
                    rows = ast.literal_eval(node.value)
                    return [{"ticker": r["code"], "name": r.get("name", ""),
                             "qty": int(r["qty"]), "cost": float(r["cost"]),
                             "kind": r.get("kind", "现物")} for r in rows]
        except Exception:
            continue
    return []


def _find_existing() -> str | None:
    """在候选目录里找已有的 portfolio.json (打包版/源码版共用同一份数据)"""
    for base in _candidate_dirs():
        p = os.path.join(base, "portfolio.json")
        if os.path.exists(p):
            return p
    return None


def load() -> list[dict]:
    path = _find_existing()
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("positions", [])
        except (OSError, json.JSONDecodeError):
            pass
    rows = _seed()          # 没有存档：从market.py导入并落盘
    if rows:
        save(rows)
    return rows


def save(positions: list[dict]) -> None:
    path = _find_existing() or PORTFOLIO_PATH    # 有存档就原地更新，避免两份数据打架
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"positions": positions}, f, ensure_ascii=False, indent=1)


# 信用交易成本(乐天证券,2026年现状)。制度信用买建的年利率;现物无利息
MARGIN_RATE_ANNUAL = 0.0280      # 買方金利 年2.80%
MARGIN_MGMT_FEE_PER_100 = 110    # 管理費 1株あたり(建玉100株ごと110円/月,简化按此计)
DAYS_PER_YEAR = 365              # 利息按实际日历天数计(不是交易日)


def holding_days(pos: dict, today=None) -> int:
    """建玉以来的日历天数。无建仓日期则返回0(利息按0算并在界面提示补录)"""
    from datetime import date, datetime as _dt
    d = pos.get("open_date")
    if not d:
        return 0
    try:
        d0 = _dt.strptime(str(d)[:10], "%Y-%m-%d").date()
    except ValueError:
        return 0
    t = today or date.today()
    return max((t - d0).days, 0)


def margin_interest(pos: dict, today=None) -> dict:
    """信用买建的持仓成本。现物返回0"""
    if pos.get("kind") != "信用买":
        return {"days": 0, "interest": 0.0, "mgmt_fee": 0.0, "total": 0.0}
    days = holding_days(pos, today)
    principal = pos["cost"] * pos["qty"]           # 建玉代金
    interest = principal * MARGIN_RATE_ANNUAL * days / DAYS_PER_YEAR
    months = days / 30.4                            # 管理费按月计
    mgmt = MARGIN_MGMT_FEE_PER_100 * (pos["qty"] / 100) * months
    return {"days": days, "interest": interest, "mgmt_fee": mgmt,
            "total": interest + mgmt}


def analyze(pos: dict, price: float, levels: dict | None, today=None) -> dict:
    """单只持仓的盈亏 + 情景测算。
    levels = trade_levels()的输出(含targets/support)，None时只给当前盈亏"""
    qty, cost = pos["qty"], pos["cost"]
    cost_amt = cost * qty
    value = price * qty
    pnl = value - cost_amt
    cost_info = margin_interest(pos, today)
    net = pnl - cost_info["total"]                  # 扣掉利息后的真实盈亏
    out = {
        "ticker": pos["ticker"], "name": pos.get("name", ""), "kind": pos.get("kind", "现物"),
        "qty": qty, "cost": cost, "price": price, "open_date": pos.get("open_date", ""),
        "cost_amt": cost_amt, "value": value, "pnl": pnl,
        "pnl_pct": pnl / cost_amt * 100 if cost_amt else 0.0,
        "hold_days": cost_info["days"],
        "interest": cost_info["interest"], "mgmt_fee": cost_info["mgmt_fee"],
        "carry_cost": cost_info["total"],
        "net_pnl": net,
        "net_pnl_pct": net / cost_amt * 100 if cost_amt else 0.0,
        "daily_carry": cost_info["total"] / cost_info["days"] if cost_info["days"] else 0.0,
        "scenarios": [],
    }
    if not levels:
        return out

    # 上行情景：到各目标位
    for t in levels.get("targets", []):
        p = t["price"]
        out["scenarios"].append({
            "name": t["name"], "kind": "上行", "price": p,
            "move_pct": (p - price) / price * 100,        # 现价还要涨多少
            "pnl": (p - cost) * qty,                      # 那时的总盈亏
            "pnl_pct": (p - cost) / cost * 100,
            "delta": (p - price) * qty,                   # 相对现在多赚多少
        })
    # 下行情景：跌到最近支撑 / 跌破支撑(止损位)
    sup = levels.get("support") or []
    if sup:
        p = sup[0]["price"]
        out["scenarios"].append({
            "name": "回踩支撑", "kind": "下行", "price": p,
            "move_pct": (p - price) / price * 100,
            "pnl": (p - cost) * qty, "pnl_pct": (p - cost) / cost * 100,
            "delta": (p - price) * qty,
        })
    stop = levels.get("stop")
    if stop:
        out["scenarios"].append({
            "name": "跌破止损", "kind": "下行", "price": stop,
            "move_pct": (stop - price) / price * 100,
            "pnl": (stop - cost) * qty, "pnl_pct": (stop - cost) / cost * 100,
            "delta": (stop - price) * qty,
        })

    ups = [s for s in out["scenarios"] if s["kind"] == "上行"]
    downs = [s for s in out["scenarios"] if s["kind"] == "下行"]
    if ups and downs:
        best_up = max(s["delta"] for s in ups)            # 最远目标的潜在增益
        worst_dn = min(s["delta"] for s in downs)         # 最差情景的潜在损失
        out["upside"] = best_up
        out["downside"] = worst_dn
        out["ratio"] = best_up / abs(worst_dn) if worst_dn else float("nan")
    return out


def totals(rows: list[dict]) -> dict:
    """组合汇总 + 各情景下的组合盈亏(情景值已扣持仓成本)"""
    t = {"cost_amt": 0.0, "value": 0.0, "pnl": 0.0, "carry_cost": 0.0, "net_pnl": 0.0,
         "T1": 0.0, "T2": 0.0, "T3": 0.0, "stop": 0.0, "daily_carry": 0.0}
    for r in rows:
        t["cost_amt"] += r["cost_amt"]
        t["value"] += r["value"]
        t["pnl"] += r["pnl"]
        t["carry_cost"] += r.get("carry_cost", 0.0)
        t["net_pnl"] += r.get("net_pnl", r["pnl"])
        t["daily_carry"] += r.get("daily_carry", 0.0)
        by = {s["name"]: s for s in r.get("scenarios", [])}
        carry = r.get("carry_cost", 0.0)
        for k in ("T1", "T2", "T3"):
            t[k] += (by[k]["pnl"] if k in by else r["pnl"]) - carry
        t["stop"] += (by["跌破止损"]["pnl"] if "跌破止损" in by else r["pnl"]) - carry
    t["pnl_pct"] = t["pnl"] / t["cost_amt"] * 100 if t["cost_amt"] else 0.0
    t["net_pnl_pct"] = t["net_pnl"] / t["cost_amt"] * 100 if t["cost_amt"] else 0.0
    return t
