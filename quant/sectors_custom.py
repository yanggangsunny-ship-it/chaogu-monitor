# -*- coding: utf-8 -*-
"""自定义领域分类：JPX的33业种太粗(半导体混在"電気機器"123只里、任天堂归在"その他製品")，
这里按投资视角重新划分成用户关心的8个领域。

两种来源：
  · curated  = 手工整理的核心标的(细分领域用,如半导体/游戏/视频)
  · jpx      = 直接用JPX 33业种(粗分够用的,如银行/机械/航运)
"""
from __future__ import annotations

from universe import load_universe

# 手工整理(代码已逐一对照JPX名录验证)
CURATED = {
    "半导体": [
        "8035", "6857", "6146", "7735", "6920", "4063", "3436", "4186", "6723", "6963",
        "285A", "6254", "7729", "6981", "6762", "4062", "6526", "6871", "6315",
        "7741", "4980", "6976", "5214", "6266",
    ],
    "游戏": [
        "7974", "9697", "9684", "9766", "7832", "3659", "2432", "3765", "3668",
        "4751", "6460", "2121",
    ],
    "娱乐": [
        "4661", "9602", "9601", "9605", "7867", "8136", "6758", "9468",
    ],
    "视频": [
        "9404", "9401", "9409", "4676", "9412", "4839", "2371",
    ],
    "电气设备": [
        "6501", "6503", "6504", "6506", "6752", "6954", "6645", "6861", "6594",
        "6841", "6702", "6701", "6479", "6367",
    ],
}

# 直接用JPX 33业种的领域
JPX_MAP = {
    "银行": "銀行業",
    "机械": "機械",
    "航运": "海運業",
}

# 专栏显示顺序(用户指定)
ORDER = ["游戏", "半导体", "电气设备", "银行", "娱乐", "视频", "机械", "航运"]


def build_sectors(market: str = "prime") -> dict[str, list[str]]:
    """返回 {领域: [ticker,...]}，自动剔除不在股票池里的代码"""
    uni = load_universe((market,))
    valid = set(uni["ticker"])
    by_jpx = {}
    for name, jpx_name in JPX_MAP.items():
        by_jpx[name] = uni.loc[uni["sector33"] == jpx_name, "ticker"].tolist()

    out = {}
    for name in ORDER:
        if name in CURATED:
            tickers = [f"{c}.T" for c in CURATED[name]]
            out[name] = [t for t in tickers if t in valid]
        else:
            out[name] = by_jpx.get(name, [])
    return out


if __name__ == "__main__":
    s = build_sectors()
    uni = load_universe(("prime",)).set_index("ticker")
    for k, v in s.items():
        names = [uni.loc[t, "name"] for t in v[:4]]
        print(f"{k:<8} {len(v):>3}只  例: {' / '.join(names)}")
    # 报告无效代码
    valid = set(uni.index)
    for k, codes in CURATED.items():
        bad = [c for c in codes if f"{c}.T" not in valid]
        if bad:
            print(f"  ⚠{k} 无效代码: {bad}")
