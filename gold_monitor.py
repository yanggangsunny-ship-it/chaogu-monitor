# -*- coding: utf-8 -*-
"""沪金Au99.99(上金所)价格监控 — 阿里云服务器独立部署版

参考 market.py(日股监控)的成熟套路：异动分档 + 开盘/收盘播报 + state持久化。
- 数据源=东方财富(国内直连,无需key,UA伪装即可)；只用标准库,零pip依赖
- 推送=Server酱多SendKey：sendkeys.txt一行一个key,同时通报到多个人的微信,单人失败互不影响
- 触发=服务器crontab粗筛(UTC 0-8点工作日每5分钟) + 脚本内细筛(北京时间/行情时间戳)
- 休市判定不查节假日表：行情时间戳(f86)不是当天(北京时间)即视为休市静默,周末/中国法定假日全覆盖

运行方式:
    python3 gold_monitor.py          单轮检查(供crontab调用)
    python3 gold_monitor.py --test   只打印推送段落,不真发
部署位置: 阿里云47.116.21.78:/root/gold/ (本仓库中的副本仅作备份,改完需scp覆盖服务器)
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

# ---------- 配置区 ----------
GOLD_NAME = "沪金Au99.99"
SECID = "118.AU9999"           # 东方财富代码：118=上金所市场
THRESHOLD = 1.5                # 涨跌破1.5%推第1档(黄金日波动通常<1%,1.5%已是显著行情)
ALERT_STEP = 1.0               # 破阈值后,涨跌幅每再扩大1%再推一档(涨跌方向独立;股票用2%对黄金太宽)
MA_WINDOW = 20                 # 均线/平均成交量天数
VOLUME_HIGH_RATIO = 1.5        # 成交量达20日均量N倍算放量
VOLUME_LOW_RATIO = 0.5         # 不足N倍算缩量

CST = timezone(timedelta(hours=8))   # 全脚本北京时间
OPEN_MIN = 9 * 60                    # 上金所日盘开盘 9:00
CLOSE_MIN = 15 * 60 + 30             # 日盘收盘 15:30 (Au99.99无夜盘)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENDKEY_FILE = os.path.join(BASE_DIR, "sendkeys.txt")        # 一行一个SendKey;600权限,不进git
STATE_FILE = os.path.join(BASE_DIR, "gold_state.json")       # 文件名带gold_前缀,避免与market.py的state.json冲突
CSV_FILE = os.path.join(BASE_DIR, "gold_price_history.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
QUOTE_URL = (
    "https://push2.eastmoney.com/api/qt/stock/get"
    f"?secid={SECID}&fields=f43,f44,f45,f46,f47,f57,f58,f60,f86,f170"
)
KLINE_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    f"?secid={SECID}&klt=101&fqt=0&lmt=260&end=20500101"
    "&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56"
)


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_gold_price():
    """东财实时行情。价格字段是×100的整数(91000=910.00元);f170=涨跌幅×100;f86=行情unix时间戳"""
    d = _get_json(QUOTE_URL)["data"]

    def _p(field):  # 价格字段还原两位小数;无数据时东财返回"-"
        v = d.get(field)
        return v / 100 if isinstance(v, (int, float)) else None

    return {
        "price": _p("f43"),
        "day_high": _p("f44"),
        "day_low": _p("f45"),
        "day_open": _p("f46"),
        "prev_close": _p("f60"),
        "pct": _p("f170"),                      # f170同样×100,-11 → -0.11%
        "volume": d.get("f47") if isinstance(d.get("f47"), (int, float)) else None,
        "quote_time": datetime.fromtimestamp(d["f86"], CST).strftime("%Y-%m-%d %H:%M")
        if d.get("f86") else None,
    }


def get_history_stats():
    """日K约260根 → 20日均价/20日均量/52周高低(带日期)。行格式: 日期,开,收,高,低,量"""
    rows = [k.split(",") for k in _get_json(KLINE_URL)["data"]["klines"]]
    closes = [float(r[2]) for r in rows]
    vols = [float(r[5]) for r in rows]
    highs = [(float(r[3]), r[0]) for r in rows]
    lows = [(float(r[4]), r[0]) for r in rows]
    high, high_date = max(highs, key=lambda x: x[0])
    low, low_date = min(lows, key=lambda x: x[0])
    # 最后一根是当天进行中的K线,均线/均量用它之前的完整交易日算
    ma = round(sum(closes[-MA_WINDOW - 1:-1]) / MA_WINDOW, 2) if len(closes) > MA_WINDOW else None
    avg_volume = round(sum(vols[-MA_WINDOW - 1:-1]) / MA_WINDOW) if len(vols) > MA_WINDOW else None
    return {"ma": ma, "avg_volume": avg_volume,
            "high": high, "high_date": high_date, "low": low, "low_date": low_date}


def _fmt(value):
    """元/克计价保留2位小数(金价0.5%波动才4.5元,不能像円那样去小数)"""
    return f"{value:,.2f}"


def _quote_time_disp(quote_time):
    """行情时刻显示：当天只显示HH:MM,非当天带日期提示是旧行情"""
    if not quote_time:
        return "?"
    today = datetime.now(CST).strftime("%Y-%m-%d")
    if quote_time.startswith(today):
        return quote_time[11:]
    return quote_time[5:]  # MM-DD HH:MM


def _volume_line(volume, avg_volume):
    if not volume:
        return "成交量: 暂无数据"
    base = f"成交量: {volume:,} 克"
    if not avg_volume:
        return base
    ratio = volume / avg_volume
    if ratio >= VOLUME_HIGH_RATIO:
        tag = "放量"
    elif ratio <= VOLUME_LOW_RATIO:
        tag = "缩量"
    else:
        tag = "量能正常"
    return f"{base} (为{MA_WINDOW}日均量{avg_volume:,}的{ratio:.1f}倍，{tag})"


def _ma_line(price, ma):
    if ma is None:
        return f"{MA_WINDOW}日均线: 数据不足"
    rel = "高于" if price >= ma else "低于"
    diff_pct = round((price - ma) / ma * 100, 2)
    line = f"{MA_WINDOW}日均线: {_fmt(ma)}元 (现价{rel}均线{abs(diff_pct)}%)"
    if price >= ma:
        line = f"🟢{line}"
    return line


def _gold_block(data, stats):
    """组装推送正文段落。涨🔴跌🟢色点(Server酱转义HTML,只能emoji)"""
    pct = data["pct"] or 0
    direction = "涨" if pct > 0 else "跌"
    if pct > 0:
        name = f"🔴{GOLD_NAME}"
    elif pct < 0:
        name = f"🟢{GOLD_NAME}"
    else:
        name = GOLD_NAME

    lines = [
        f"### {name}",
        f"{direction}{abs(pct)}% | 现价 **{_fmt(data['price'])}** 元/克@{_quote_time_disp(data.get('quote_time'))}"
        f" (昨收 {_fmt(data['prev_close'])})",
    ]
    if stats:
        lines.append(_volume_line(data["volume"], stats["avg_volume"]))
    else:
        lines.append(_volume_line(data["volume"], None))
    if data["day_low"] is not None and data["day_high"] is not None:
        lines.append(f"当日: {_fmt(data['day_low'])} ~ {_fmt(data['day_high'])} 元/克 (今开 {_fmt(data['day_open'])})")
    if stats:
        lines.append(_ma_line(data["price"], stats["ma"]))
        lines.append(f"52周: {_fmt(stats['low'])}元({stats['low_date']}) ~ {_fmt(stats['high'])}元({stats['high_date']})")
    return "\n\n".join(lines)


def log_price(data):
    """行情追加写CSV,保留历史供复盘(格式对齐market.py的log_price)"""
    is_new = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["抓取时间", "行情时刻", "价格", "昨收", "涨跌幅%", "成交量", "当日最高", "当日最低"])
        writer.writerow([
            datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S"),
            data.get("quote_time") or "",
            data["price"], data["prev_close"], data["pct"],
            data["volume"], data["day_high"], data["day_low"],
        ])


# ---------- 状态持久化(单轮模式跨运行保留,结构对齐market.py) ----------
def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        loaded = {}
    return {
        "last_open_date": loaded.get("last_open_date"),
        "last_close_date": loaded.get("last_close_date"),
        "alert_date": loaded.get("alert_date"),
        "up_tier": loaded.get("up_tier", 0),
        "down_tier": loaded.get("down_tier", 0),
    }


def save_state(st):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def _tier(pct_abs):
    """涨跌幅达到第几档：<阈值=0档;[阈值,阈值+STEP)=1档;以此类推"""
    if pct_abs < THRESHOLD:
        return 0
    return 1 + int((pct_abs - THRESHOLD) // ALERT_STEP)


# ---------- 推送(Server酱多人) ----------
def _load_sendkeys():
    try:
        with open(SENDKEY_FILE, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    except FileNotFoundError:
        return []


def send_all(title, desp):
    """逐个SendKey推送,单人失败不影响其他人"""
    keys = _load_sendkeys()
    if not keys:
        print(f"[推送] 没有可用SendKey({SENDKEY_FILE}),跳过: {title}")
        return
    payload = urllib.parse.urlencode({"title": title, "desp": desp}).encode("utf-8")
    for i, key in enumerate(keys, 1):
        try:
            req = urllib.request.Request(f"https://sctapi.ftqq.com/{key}.send", data=payload, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") == 0:
                print(f"[推送] 第{i}人 OK: {title}")
            else:
                print(f"[推送] 第{i}人 失败: {result}")
        except Exception as e:
            print(f"[推送] 第{i}人 出错(不影响其他人): {e}")


# ---------- 主流程 ----------
def check_once():
    now = datetime.now(CST)
    today = now.strftime("%Y-%m-%d")
    t = now.hour * 60 + now.minute

    if now.weekday() >= 5:  # 周末先挡一道,省一次请求
        print(f"{now.strftime('%H:%M')} 周末休市")
        return

    data = get_gold_price()
    # 休市判定：行情时间戳不是当天 → 中国节假日/尚未开盘,全静默(不推旧数据)
    if not data.get("quote_time") or not data["quote_time"].startswith(today):
        print(f"{now.strftime('%H:%M')} 行情非当天({data.get('quote_time')}),休市或未开盘,跳过")
        return

    log_price(data)
    try:
        stats = get_history_stats()
    except Exception as e:
        print(f"历史K线获取失败(段落降级,不影响提醒): {e}")
        stats = None

    st = load_state()
    events = []  # [(reason, ...)]

    if t >= OPEN_MIN and st["last_open_date"] != today:
        events.append("开盘")
        st["last_open_date"] = today
    if t >= CLOSE_MIN and st["last_close_date"] != today:
        events.append("收盘")
        st["last_close_date"] = today

    trading = OPEN_MIN <= t <= CLOSE_MIN
    if trading:
        print(f"{now.strftime('%H:%M')} {GOLD_NAME} {data['price']} 元/克, 涨跌 {data['pct']}%")
        if st["alert_date"] != today:  # 新的一天,涨跌分档重置
            st["alert_date"] = today
            st["up_tier"] = 0
            st["down_tier"] = 0
        pct = data["pct"] or 0
        if pct >= 0:
            tier = _tier(pct)
            if tier > st["up_tier"]:
                events.append("异动")
                st["up_tier"] = tier
        else:
            tier = _tier(-pct)
            if tier > st["down_tier"]:
                events.append("异动")
                st["down_tier"] = tier
    else:
        print(f"{now.strftime('%H:%M')} 非交易时段")

    block = _gold_block(data, stats) if events else None
    footer = f"\n\n推送生成: {now.strftime('%Y-%m-%d %H:%M')} (北京时间)"
    for reason in events:
        pct = data["pct"] or 0
        if reason == "异动":
            direction = "涨" if pct >= 0 else "跌"
            title = f"沪金异动 {direction}{abs(pct)}%"
        else:
            title = f"沪金{reason}播报"
        send_all(title, block + footer)

    save_state(st)


def main():
    if "--test" in sys.argv:
        data = get_gold_price()
        try:
            stats = get_history_stats()
        except Exception as e:
            print(f"历史K线获取失败: {e}")
            stats = None
        print(_gold_block(data, stats))
        print(f"\n[test] SendKey共{len(_load_sendkeys())}个(不真发)")
        return
    check_once()


if __name__ == "__main__":
    main()
