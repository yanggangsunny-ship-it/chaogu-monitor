# -*- coding: utf-8 -*-
"""收藏股票：本地持久化的关注列表(exe旁的json,换机拷贝即可迁移)"""
from __future__ import annotations

import json
import os
import sys


def _base_dir() -> str:
    """打包后用exe所在目录，源码运行用脚本目录 —— 保证收藏跟着程序走"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


WATCH_PATH = os.path.join(_base_dir(), "watchlist.json")


def load() -> list[str]:
    try:
        with open(WATCH_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [str(t) for t in data.get("tickers", [])]
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


def save(tickers: list[str]) -> None:
    seen, out = set(), []
    for t in tickers:            # 去重保序
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    with open(WATCH_PATH, "w", encoding="utf-8") as f:
        json.dump({"tickers": out}, f, ensure_ascii=False, indent=1)


def add(ticker: str) -> list[str]:
    lst = load()
    if ticker not in lst:
        lst.append(ticker)
        save(lst)
    return lst


def remove(ticker: str) -> list[str]:
    lst = [t for t in load() if t != ticker]
    save(lst)
    return lst


def contains(ticker: str) -> bool:
    return ticker in load()
