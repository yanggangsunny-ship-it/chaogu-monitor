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

# 首次运行时的种子数据：从监控系统 market.py 的 POSITIONS 导入
_SEED_FROM_MARKET = os.path.join(os.path.dirname(_base_dir()), "market.py")


def _seed() -> list[dict]:
    """尝试从 chaogu/market.py 读取现有持仓(只读，不修改那边)"""
    try:
        import ast
        src = open(_SEED_FROM_MARKET, encoding="utf-8").read()
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "POSITIONS" for t in node.targets):
                rows = ast.literal_eval(node.value)
                return [{"ticker": r["code"], "name": r.get("name", ""),
                         "qty": int(r["qty"]), "cost": float(r["cost"]),
                         "kind": r.get("kind", "现物")} for r in rows]
    except Exception:
        pass
    return []


def load() -> list[dict]:
    if os.path.exists(PORTFOLIO_PATH):
        try:
            with open(PORTFOLIO_PATH, encoding="utf-8") as f:
                return json.load(f).get("positions", [])
        except (OSError, json.JSONDecodeError):
            return []
    rows = _seed()          # 首次运行：从market.py导入并落盘
    if rows:
        save(rows)
    return rows


def save(positions: list[dict]) -> None:
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump({"positions": positions}, f, ensure_ascii=False, indent=1)


def analyze(pos: dict, price: float, levels: dict | None) -> dict:
    """单只持仓的盈亏 + 情景测算。
    levels = trade_levels()的输出(含targets/support)，None时只给当前盈亏"""
    qty, cost = pos["qty"], pos["cost"]
    cost_amt = cost * qty
    value = price * qty
    pnl = value - cost_amt
    out = {
        "ticker": pos["ticker"], "name": pos.get("name", ""), "kind": pos.get("kind", "现物"),
        "qty": qty, "cost": cost, "price": price,
        "cost_amt": cost_amt, "value": value, "pnl": pnl,
        "pnl_pct": pnl / cost_amt * 100 if cost_amt else 0.0,
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
    """组合汇总 + 各情景下的组合盈亏"""
    t = {"cost_amt": 0.0, "value": 0.0, "pnl": 0.0,
         "T1": 0.0, "T2": 0.0, "T3": 0.0, "stop": 0.0}
    for r in rows:
        t["cost_amt"] += r["cost_amt"]
        t["value"] += r["value"]
        t["pnl"] += r["pnl"]
        by = {s["name"]: s for s in r.get("scenarios", [])}
        for k in ("T1", "T2", "T3"):
            t[k] += by[k]["pnl"] if k in by else r["pnl"]      # 无该目标则按现状计
        t["stop"] += by["跌破止损"]["pnl"] if "跌破止损" in by else r["pnl"]
    t["pnl_pct"] = t["pnl"] / t["cost_amt"] * 100 if t["cost_amt"] else 0.0
    return t
