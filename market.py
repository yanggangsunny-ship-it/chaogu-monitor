# -*- coding: utf-8 -*-
"""
日股股价异动监控脚本(支持多只股票)
逻辑: 东证交易时段每10分钟检查一次 → 涨跌超阈值推送微信(Server酱)；破阈值后每再扩大ALERT_STEP又推一次(涨跌方向独立分档，每天重置)
      另外每天开盘(9:00)、收盘(15:30)后各推送一次当前行情，同样当天只推一次
      同一轮检查里，多只股票触发的开盘/收盘/异动分别合并成一条推送(不是每只股票单独发)
      每只股票独立记录历史价格(CSV)并自动重画走势图(PNG)
运行: python market.py          常驻模式(本机挂机,保持窗口开着)
      python market.py --once   单轮模式(GitHub Actions定时用,查一轮就退出,状态存state.json)
"""

import os
import csv
import sys
import json
import re
import io
import zipfile
import smtplib
import requests
import time
import difflib
from email.mime.text import MIMEText
from email.header import Header

try:
    import jpholiday   # 日本法定节假日历
except ImportError:
    jpholiday = None   # 未安装时退化为只判断周末(不至于崩)
import xml.etree.ElementTree as ET
from urllib.parse import quote
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============ 配置区（只需要改这里） ============
def _load_sendkey():
    """Server酱SendKey: 优先环境变量(GitHub Actions用Secrets注入)，其次脚本旁的sendkey.txt(本机用,不入git)"""
    key = os.environ.get("SENDKEY")
    if key:
        return key.strip()
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sendkey.txt"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        print("警告: 未配置SendKey(环境变量SENDKEY或sendkey.txt)，推送将失败")
        return ""


def _load_secret_file(env_name, filename):
    """密钥读取：优先环境变量(Actions Secrets)，其次脚本旁的本地文件(不入git)；没有返回空串"""
    val = os.environ.get(env_name)
    if val:
        return val.strip()
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


SENDKEY = _load_sendkey()
# Gmail推送(可选,与微信并行)：GMAIL_APP_PASSWORD=发件账号的应用专用密码(myaccount.google.com/apppasswords)
# 未配置密码则自动跳过邮件只发微信；发件账号与收件箱可以不同
GMAIL_USER = os.environ.get("GMAIL_USER", "yanggang.sunny@gmail.com")   # 发件账号(应用密码所属)
MAIL_TO = os.environ.get("MAIL_TO", "yanggang.sunny@gmail.com")         # 收件箱
GMAIL_APP_PASSWORD = _load_secret_file("GMAIL_APP_PASSWORD", "gmail_pass.txt")
# LLM新闻分析(可选)：Anthropic API key；未配置则新闻退化为原始日文标题
ANTHROPIC_API_KEY = _load_secret_file("ANTHROPIC_API_KEY", "anthropic_key.txt")
LLM_MODEL = "claude-haiku-4-5-20251001"          # 新闻摘要/打分用轻量模型,成本每月$1~2
NEWS_RELEVANCE_MIN = 5                           # LLM相关度(0-10)低于此分的新闻直接丢弃(降噪核心)
CHECK_INTERVAL = 600                             # 检查间隔(秒)，600 = 10分钟
NEWS_COUNT = 3                                    # 附带新闻条数(去重后)
NEWS_DEDUP_THRESHOLD = 0.35                       # 标题相似度超过此值视为同一事件，只保留最新一条
ALERT_STEP = 2.0                                  # 破阈值后，涨跌幅每再扩大这么多再推一次(涨跌方向独立计算)
MA_WINDOW = 20                                    # 均线/平均成交量的天数
VOLUME_HIGH_RATIO = 1.5                           # 成交量达到N倍20日均量算放量
VOLUME_LOW_RATIO = 0.5                            # 成交量不到N倍20日均量算缩量
EARNINGS_REMIND_BDAYS = 2                         # 财报发布前N个工作日推送提醒
TDNET_RECENT_URL = "https://webapi.yanoshin.jp/webapi/tdnet/list/recent.json?limit=100"  # TDnet披露流(非官方JSON镜像)

# 要监控的股票列表，每只一个字典；code用Yahoo Finance代码(日股格式如 9984.T，指数如 ^N225)
STOCKS = [
    {"code": "^N225", "name": "日经225指数", "threshold": 2.0, "news_query": "日経平均株価"},
    {"code": "9984.T", "name": "软银集团 (9984.T)", "threshold": 3.0, "news_query": "ソフトバンクグループ"},
    {"code": "285A.T", "name": "铠侠控股 (285A)", "threshold": 3.0, "news_query": "キオクシア"},
    {"code": "7012.T", "name": "川崎重工 (7012)", "threshold": 3.0, "news_query": "川崎重工業"},
    {"code": "7011.T", "name": "三菱重工 (7011)", "threshold": 3.0, "news_query": "三菱重工業"},
    {"code": "6954.T", "name": "发那科 (6954)", "threshold": 3.0, "news_query": "ファナック"},
    {"code": "6264.T", "name": "Marumae (6264)", "threshold": 3.0, "news_query": "マルマエ"},
    {"code": "4063.T", "name": "信越化学 (4063)", "threshold": 3.0, "news_query": "信越化学工業"},
    {"code": "9697.T", "name": "卡普空 (9697)", "threshold": 3.0, "news_query": "カプコン"},
    {"code": "9501.T", "name": "东京电力 (9501)", "threshold": 3.0, "news_query": "東京電力"},
    {"code": "8306.T", "name": "三菱UFJ金融集团 (8306)", "threshold": 3.0, "news_query": "三菱UFJフィナンシャル・グループ"},
    {"code": "7186.T", "name": "横滨金融集团 (7186)", "threshold": 3.0, "news_query": "横浜フィナンシャルグループ"},
    {"code": "4751.T", "name": "CyberAgent (4751)", "threshold": 3.0, "news_query": "サイバーエージェント"},
    {"code": "6758.T", "name": "索尼集团 (6758)", "threshold": 3.0, "news_query": "ソニーグループ"},
    {"code": "7974.T", "name": "任天堂 (7974)", "threshold": 3.0, "news_query": "任天堂"},
    {"code": "1579.T", "name": "日经Bull2倍ETF (1579)", "threshold": 4.0, "news_query": "日経平均株価"},  # 2倍杠杆,4%≈日经2%
    {"code": "8593.T", "name": "三菱HC Capital (8593)", "threshold": 3.0, "news_query": "三菱HCキャピタル"},
]

# 持仓列表(2026-07-04截图录入)：现物=平均取得价，信用买建=建单价；开盘/收盘推送末尾附盈亏报告
POSITIONS = [
    {"code": "1579.T", "name": "日经Bull2倍ETF (1579)", "qty": 1200, "cost": 831.53, "kind": "现物"},
    # 已平仓: 4751 CyberAgent 现物100株@1,379 → 2026-07-09 @1,534卖出, 实现损益 +15,500円(+11.2%)
    {"code": "7186.T", "name": "横滨金融集团 (7186)", "qty": 100, "cost": 1781.50, "kind": "现物"},
    {"code": "8306.T", "name": "三菱UFJ (8306)", "qty": 100, "cost": 3135.00, "kind": "现物"},
    {"code": "9501.T", "name": "东京电力 (9501)", "qty": 100, "cost": 689.00, "kind": "现物"},
    {"code": "4063.T", "name": "信越化学 (4063)", "qty": 100, "cost": 7155.0, "kind": "信用买"},
    # 已平仓: 6264 Marumae 信用买100株@2,793 → 2026-07-07 @2,330卖出, 实现损益 -46,300円(-16.6%)
    {"code": "6954.T", "name": "发那科 (6954)", "qty": 100, "cost": 8148.9, "kind": "信用买"},
    {"code": "7011.T", "name": "三菱重工 (7011)", "qty": 100, "cost": 4800.0, "kind": "信用买"},
    {"code": "7012.T", "name": "川崎重工 (7012)", "qty": 100, "cost": 2859.5, "kind": "信用买"},
    {"code": "8593.T", "name": "三菱HC Capital (8593)", "qty": 100, "cost": 1447.0, "kind": "信用买"},
    {"code": "9984.T", "name": "软银集团 (9984.T)", "qty": 100, "cost": 6500.0, "kind": "信用买"},
]

# 信用交易保证金监控(乐天证券规则)。维持率=(受入保证金-建玉评价损)/建玉代金
# 受入保证金=现金+代用有价证券评价额×掛目,会随现物担保波动;此处用基准值动态估算维持率,
# ⚠偏乐观(未跟踪代用担保缩水),需用户定期用券商真实维持率校准 MARGIN_DEPOSIT
MARGIN_DEPOSIT = 1338696   # 受入保证金基准(2026-07-09由维持率32.41%反推;现物担保约123万+现金约11万)
MARGIN_CALL_LINE = 20      # 乐天追证线:维持率<20%→翌々营业日12:00前补钱或平仓
MARGIN_FORCE_LINE = 10     # 乐天强制平仓线:<10%券商无需追证直接砍建玉
MARGIN_WARN_LINE = 28      # 自定预警线:维持率≤此值收盘播报标红警告(用户当前32.4%已贴近建仓线30%,预警从早)

# 潜力股筛选：6大领域股票池(共110只,均已验证有效)。每天收盘后先按当日成交额取每领域前SECTOR_TOP_N只，
# 再对入选股跑走势条件；SCREEN_EXTRA=不参与排名、每天必扫的自选(如持仓ETF)
SECTOR_TOP_N = 20
SECTOR_UNIVERSE = {
    "半导体/AI": ["8035.T", "6857.T", "6146.T", "7735.T", "6920.T", "4063.T", "3436.T", "4186.T",
                  "6723.T", "6963.T", "285A.T", "6254.T", "7729.T", "6981.T", "6762.T", "4062.T",
                  "6526.T", "6594.T", "6890.T", "6871.T", "6315.T", "7741.T", "4980.T", "6976.T",
                  "5214.T", "6266.T", "6264.T", "9984.T", "6954.T", "6506.T"],
    "金融": ["8306.T", "8316.T", "8411.T", "8308.T", "8309.T", "7186.T", "8331.T", "5831.T",
             "8354.T", "7167.T", "7180.T", "8591.T", "8593.T", "8697.T", "8604.T", "8601.T",
             "8630.T", "8725.T", "8766.T", "8750.T", "8795.T", "7181.T", "7182.T", "8253.T", "8572.T"],
    "防卫": ["7011.T", "7012.T", "7013.T", "5631.T", "6203.T", "7721.T", "4274.T", "6701.T",
             "6702.T", "6946.T"],
    "电力/电网": ["9501.T", "9502.T", "9503.T", "9504.T", "9505.T", "9506.T", "9507.T", "9508.T",
                  "9509.T", "9513.T", "6501.T", "6503.T", "6504.T", "6841.T", "5801.T", "5802.T",
                  "5803.T", "1963.T", "6366.T", "9531.T", "9532.T"],
    "游戏/内容": ["7974.T", "9697.T", "6758.T", "7832.T", "9684.T", "9766.T", "3659.T", "2432.T",
                  "3765.T", "4751.T", "4816.T", "9468.T", "6460.T", "4661.T", "9602.T", "9601.T", "7867.T"],
    "商社": ["8058.T", "8031.T", "8001.T", "8002.T", "8053.T", "8015.T", "2768.T"],
}
SCREEN_EXTRA = ["1579.T"]  # 持仓ETF等,不排名每天必扫

# 筛选池中文名(推送显示用,Yahoo只有英文名;易混淆的加了注记)
CN_NAMES = {
    # 半导体/AI
    "8035.T": "东京电子", "6857.T": "爱德万测试", "6146.T": "迪思科Disco", "7735.T": "SCREEN",
    "6920.T": "Lasertec", "4063.T": "信越化学", "3436.T": "胜高SUMCO", "4186.T": "东京应化",
    "6723.T": "瑞萨电子", "6963.T": "罗姆ROHM", "285A.T": "铠侠控股", "6254.T": "野村微科学",
    "7729.T": "东京精密", "6981.T": "村田制作所", "6762.T": "TDK", "4062.T": "揖斐电IBIDEN",
    "6526.T": "索思未来Socionext", "6594.T": "尼得科Nidec", "6890.T": "Ferrotec", "6871.T": "日本麦克隆尼",
    "6315.T": "TOWA", "7741.T": "豪雅HOYA", "4980.T": "迪睿合Dexerials", "6976.T": "太阳诱电",
    "5214.T": "日本电气硝子", "6266.T": "Tazmo", "6264.T": "Marumae", "9984.T": "软银集团",
    "6954.T": "发那科", "6506.T": "安川电机",
    # 金融
    "8306.T": "三菱UFJ", "8316.T": "三井住友金融", "8411.T": "瑞穗金融", "8308.T": "里索那控股",
    "8309.T": "三井住友信托", "7186.T": "横滨金融", "8331.T": "千叶银行", "5831.T": "静冈金融",
    "8354.T": "福冈金融", "7167.T": "Mebuki金融", "7180.T": "九州金融", "8591.T": "欧力士ORIX",
    "8593.T": "三菱HC Capital", "8697.T": "日本交易所JPX", "8604.T": "野村控股", "8601.T": "大和证券",
    "8630.T": "SOMPO损保", "8725.T": "三井住友海上", "8766.T": "东京海上", "8750.T": "第一生命",
    "8795.T": "T&D控股", "7181.T": "简保生命", "7182.T": "日本邮储银行", "8253.T": "Credit Saison",
    "8572.T": "Acom消费金融",
    # 防卫
    "7011.T": "三菱重工", "7012.T": "川崎重工", "7013.T": "IHI", "5631.T": "日本制钢所",
    "6203.T": "丰和工业", "7721.T": "东京计器", "4274.T": "细谷火工", "6701.T": "NEC日本电气",
    "6702.T": "富士通", "6946.T": "日本Avionics",
    # 电力/电网
    "9501.T": "东京电力", "9502.T": "中部电力", "9503.T": "关西电力", "9504.T": "中国电力(日本·广岛)",
    "9505.T": "北陆电力", "9506.T": "东北电力(日本)", "9507.T": "四国电力", "9508.T": "九州电力",
    "9509.T": "北海道电力", "9513.T": "电源开发J-POWER", "6501.T": "日立制作所", "6503.T": "三菱电机",
    "6504.T": "富士电机", "6841.T": "横河电机", "5801.T": "古河电工", "5802.T": "住友电工",
    "5803.T": "藤仓Fujikura", "1963.T": "日挥控股", "6366.T": "千代田化工", "9531.T": "东京瓦斯",
    "9532.T": "大阪瓦斯",
    # 游戏/内容
    "7974.T": "任天堂", "9697.T": "卡普空", "6758.T": "索尼集团", "7832.T": "万代南梦宫",
    "9684.T": "史克威尔艾尼克斯", "9766.T": "科乐美KONAMI", "3659.T": "NEXON", "2432.T": "DeNA",
    "3765.T": "GungHo", "4751.T": "CyberAgent", "4816.T": "东映动画", "9468.T": "角川KADOKAWA",
    "6460.T": "世嘉飒美", "4661.T": "东方乐园(东京迪士尼)", "9602.T": "东宝(日本影业)", "9601.T": "松竹",
    "7867.T": "多美Takara Tomy",
    # 商社
    "8058.T": "三菱商事", "8031.T": "三井物产", "8001.T": "伊藤忠商事", "8002.T": "丸红",
    "8053.T": "住友商事", "8015.T": "丰田通商", "2768.T": "双日",
}
# ================================================

JST = timezone(timedelta(hours=9))
MARKET_OPEN_MIN = 9 * 60          # 开盘 9:00
# (开盘播报功能已于2026-07-07按用户要求取消;当日跳空缺口在收盘播报的技术信号行里体现)
MARKET_CLOSE_MIN = 15 * 60 + 30   # 收盘 15:30(用于交易时段判断)
MARKET_CLOSE_PUSH_MIN = 15 * 60 + 40  # 收盘播报/选股推送时点 15:40——15:30整点收盘集合竞价(Itayose)的
                                      # 最终价约15:30打出,Yahoo需1~2分钟才刷新;15:30就推会拿到收盘前最后连续竞价的旧tick(如15:18)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "state.json")

# Windows本机用微软雅黑；GitHub Actions的ubuntu上用Noto CJK(工作流里apt安装)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False


def _default_stock_state():
    # up_tier/down_tier: 当天涨/跌方向已经推送到第几档(0=还没破阈值)，每天随 alert_date 变化重置
    return {
        "last_open_date": None,
        "last_close_date": None,
        "alert_date": None,
        "up_tier": 0,
        "down_tier": 0,
    }


def load_state():
    """从state.json读回上次的推送状态(单轮模式跨运行保留)；没有或损坏就全新开始"""
    loaded = {}
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        pass
    st = {
        s["code"]: {**_default_stock_state(), **loaded.get(s["code"], {})}
        for s in STOCKS
    }
    # 全局状态：财报日缓存/提醒记录、已处理的TDnet披露id
    g = loaded.get("_global", {})
    st["_global"] = {
        "earnings_check_date": g.get("earnings_check_date"),
        "earnings": g.get("earnings", {}),
        "tanshin_seen": g.get("tanshin_seen", []),
        "premarket_date": g.get("premarket_date"),
        "screen_date": g.get("screen_date"),
    }
    return st


def save_state():
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# 每只股票独立的推送/记录状态，按 code 索引
state = load_state()


def _safe_code(code):
    return code.replace(".", "_").replace("^", "")


def _csv_path(code):
    return os.path.join(BASE_DIR, f"price_history_{_safe_code(code)}.csv")


def _chart_path(code):
    return os.path.join(BASE_DIR, f"price_chart_{_safe_code(code)}.png")


def is_market_closed_day(d):
    """东证休市日：周末 / 日本法定节假日(jpholiday) / 年末年始(12/31~1/3,交易所规则非法定假日)"""
    if d.weekday() >= 5:
        return True
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day <= 3):
        return True
    if jpholiday is not None and jpholiday.is_holiday(d):
        return True
    return False


def is_trading_time():
    """判断当前是否为东证交易时段(交易日 9:00-11:30, 12:30-15:30 JST)"""
    now = datetime.now(JST)
    if is_market_closed_day(now.date()):
        return False
    t = now.hour * 60 + now.minute
    morning = MARKET_OPEN_MIN <= t <= 11 * 60 + 30
    afternoon = 12 * 60 + 30 <= t <= MARKET_CLOSE_MIN
    return morning or afternoon


def get_price(code):
    """从Yahoo Finance获取价格、成交量、当日振幅、52周高低"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?interval=1d&range=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=15)
    meta = resp.json()["chart"]["result"][0]["meta"]
    price = meta["regularMarketPrice"]
    prev_close = meta["chartPreviousClose"]
    pct = round((price - prev_close) / prev_close * 100, 2)
    return {
        "price": price,
        "prev_close": prev_close,
        "pct": pct,
        "volume": meta["regularMarketVolume"],
        "day_high": meta["regularMarketDayHigh"],
        "day_low": meta["regularMarketDayLow"],
        "week52_high": meta["fiftyTwoWeekHigh"],
        "week52_low": meta["fiftyTwoWeekLow"],
        # 行情本身的时间戳(交易所报价时刻,非抓取时刻)——用于对照推送延迟
        "quote_time": datetime.fromtimestamp(meta["regularMarketTime"], JST).strftime("%Y-%m-%d %H:%M")
        if meta.get("regularMarketTime") else None,
    }


def log_price(stock, data):
    """把本次抓到的行情追加写入该股票的CSV，保留完整历史供画图/复盘对比。
    注：只记单次请求零成本拿到的字段；均线/RSI等指标可从价格序列事后推算，不逐行存"""
    csv_file = _csv_path(stock["code"])
    is_new = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["抓取时间", "行情时刻", "价格", "昨收", "涨跌幅%", "成交量", "当日最高", "当日最低", "52周最高", "52周最低"])
        writer.writerow([
            datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
            data.get("quote_time") or "",
            data["price"], data["prev_close"], data["pct"],
            data["volume"], data["day_high"], data["day_low"],
            data["week52_high"], data["week52_low"],
        ])


def plot_price(stock):
    """读取该股票的历史CSV，重新画出价格走势图并覆盖保存成PNG"""
    csv_file = _csv_path(stock["code"])
    times, prices = [], []
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            times.append(datetime.strptime(row["抓取时间"], "%Y-%m-%d %H:%M:%S"))
            prices.append(float(row["价格"]))

    plt.figure(figsize=(12, 5))
    plt.plot(times, prices, marker="o", markersize=2, linewidth=1)
    plt.title(f"{stock['name']} 价格走势")
    plt.xlabel("时间")
    plt.ylabel("价格(円)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.gcf().autofmt_xdate()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_chart_path(stock["code"]))
    plt.close()


def _dedup_news(raw_items):
    """标题相似度去重：同一事件的不同来源报道只保留最先出现(最新)的一条"""
    kept_norms = []
    kept = []
    for title, link in raw_items:
        norm = title.rsplit(" - ", 1)[0]  # 去掉末尾的来源名，避免来源不同拉低相似度
        if any(difflib.SequenceMatcher(None, norm, k).ratio() > NEWS_DEDUP_THRESHOLD for k in kept_norms):
            continue
        kept_norms.append(norm)
        kept.append((title, link))
        if len(kept) >= NEWS_COUNT:
            break
    return kept


def get_history_stats(code):
    """抓一年日线数据，一次算出：52周最高/最低价及各自日期、MA_WINDOW日均线、MA_WINDOW日平均成交量"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?interval=1d&range=1y"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=15)
    result = resp.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    high_val = high_date = low_val = low_date = None
    for ts, h, l in zip(timestamps, quote["high"], quote["low"]):
        date = datetime.fromtimestamp(ts, JST).strftime("%Y-%m-%d")
        if h is not None and (high_val is None or h > high_val):
            high_val, high_date = h, date
        if l is not None and (low_val is None or l < low_val):
            low_val, low_date = l, date

    recent_closes = [c for c in quote["close"] if c is not None][-MA_WINDOW:]
    recent_volumes = [v for v in quote["volume"] if v is not None][-MA_WINDOW:]
    ma = round(sum(recent_closes) / len(recent_closes), 2) if recent_closes else None
    avg_volume = round(sum(recent_volumes) / len(recent_volumes)) if recent_volumes else None

    return {
        "high": high_val, "high_date": high_date,
        "low": low_val, "low_date": low_date,
        "ma": ma,
        "avg_volume": avg_volume,
    }


def get_intraday_extremes(code):
    """抓当日5分钟线数据，找出当日最高/最低价各自发生的时间点"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?interval=5m&range=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=15)
    result = resp.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    high_val = high_time = low_val = low_time = None
    for ts, h, l in zip(timestamps, quote["high"], quote["low"]):
        t = datetime.fromtimestamp(ts, JST).strftime("%H:%M")
        if h is not None and (high_val is None or h > high_val):
            high_val, high_time = h, t
        if l is not None and (low_val is None or l < low_val):
            low_val, low_time = l, t

    return {"high": high_val, "high_time": high_time, "low": low_val, "low_time": low_time}


def _llm_call(system, user_msg, max_tokens=1000):
    """调用Anthropic API(直接HTTP,不引SDK)；未配置key返回None"""
    if not ANTHROPIC_API_KEY:
        return None
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": LLM_MODEL, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user_msg}]},
        timeout=45,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def analyze_news_llm(stock_name, news):
    """LLM新闻分析：中文摘要+利好/利空/中性+相关度打分,低相关的丢弃(降噪)。
    返回[(摘要,立场,相关度,链接)]按相关度降序；未配置key或失败返回None(调用方回退原标题)"""
    if not ANTHROPIC_API_KEY or not news:
        return None
    numbered = "\n".join(f"{i + 1}. {t}" for i, (t, _) in enumerate(news))
    system = (
        "你是日股新闻分析师。对给出的每条新闻标题输出一个JSON对象:"
        '{"i":序号,"s":"中文摘要一句话25字内","j":"利好|利空|中性","r":相关度整数0到10}。'
        "r=该新闻对这只股票股价的影响相关程度：业绩/订单/投资/事故/监管/评级=高分；"
        "泛泛提及/行情复述/榜单/广告软文=低分。j从股价角度判断。只输出JSON数组,无其他文字。"
    )
    text = _llm_call(system, f"股票: {stock_name}\n新闻标题:\n{numbered}", max_tokens=800)
    if not text:
        return None
    m = re.search(r"\[.*\]", text, re.S)
    items = json.loads(m.group(0) if m else text)
    out = []
    for it in items:
        idx = int(it["i"]) - 1
        if 0 <= idx < len(news) and int(it.get("r", 0)) >= NEWS_RELEVANCE_MIN:
            out.append((str(it["s"]), str(it["j"]), int(it["r"]), news[idx][1]))
    out.sort(key=lambda x: -x[2])
    return out


def _html_to_text(html):
    """粗提取HTML正文文本(供LLM读,无需精细排版)"""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip()


def analyze_tanshin_quality(zip_bytes, stock_name):
    """财报科目异常预筛：把决算短信的定性说明(qualitative.htm)交给LLM按检查清单过一遍——
    一次性损益/减损拨备/会计政策变更/汇率影响/增长是否来自主业。返回'科目体检'文本行或None"""
    if not ANTHROPIC_API_KEY or not zip_bytes:
        return None
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    qual_files = [n for n in z.namelist() if "qualitative" in n.lower()]
    if not qual_files:
        return None
    text = _html_to_text(z.read(qual_files[0]).decode("utf-8", errors="ignore"))[:8000]
    if len(text) < 100:
        return None
    system = (
        "你是财务分析师,审阅日本上市公司决算短信的定性说明,按检查清单排查'盈利成色'问题:"
        "①一次性损益(特别利益/损失、补助金、资产出售、保险金收入)②减损损失/大额拨备"
        "③会计政策或折旧方法变更④汇率对损益的重大影响⑤业绩增减是否来自主业本身"
        "⑥若有业绩预想大幅变化,其真实原因。"
        '输出JSON:{"findings":[{"t":"类型简称","d":"说明,30字以内"}],"grade":"高|中|低","note":"盈利成色一句话总评,30字以内"}。'
        "findings只列文中确实存在的问题,没有就用空数组;grade=盈利成色(高=主业驱动干净/低=依赖一次性因素)。"
        "所有输出必须用简体中文(金额可保留原数字,公司名可保留原文),严禁输出日文句子。只输出JSON。"
    )
    out = _llm_call(system, f"公司: {stock_name}\n决算短信定性说明:\n{text}", max_tokens=600)
    if not out:
        return None
    m = re.search(r"\{.*\}", out, re.S)
    data = json.loads(m.group(0) if m else out)
    findings = data.get("findings", [])[:5]
    grade = data.get("grade", "?")
    note = data.get("note", "")
    if findings:
        body = " | ".join(f"⚠{fd.get('t', '?')}: {fd.get('d', '')}" for fd in findings)
        return f"科目体检(盈利成色:{grade}): {body}" + (f"。{note}" if note else "")
    return f"科目体检: 未见一次性损益/会计变更等异常项(盈利成色:{grade})" + (f"。{note}" if note else "")


def get_news(query):
    """从Google News RSS抓取相关新闻标题，去重后最多NEWS_COUNT条，用于异动推送时附带消息面"""
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=ja&gl=JP&ceid=JP:ja"
    resp = requests.get(url, timeout=10)
    root = ET.fromstring(resp.content)
    items = root.findall("./channel/item")[:NEWS_COUNT * 4]  # 多取一些候选，去重后再截断
    raw = [(item.findtext("title"), item.findtext("link")) for item in items]
    return _dedup_news(raw)


def _yen(value):
    """円计价数字去掉小数点，省文本量"""
    return f"{round(value):,}"


def _quote_time_disp(quote_time):
    """行情时刻显示：当天只显示HH:MM，非当天(周末/休市)带日期提示是旧行情"""
    if not quote_time:
        return "?"
    today = datetime.now(JST).strftime("%Y-%m-%d")
    if quote_time.startswith(today):
        return quote_time[11:]
    return quote_time[5:]  # MM-DD HH:MM


def _ma_line(price, ma):
    """20日均线：现价相对均线的位置，辅助判断趋势；现价高于均线时整行标绿"""
    if ma is None:
        return f"{MA_WINDOW}日均线: 数据不足"
    rel = "高于" if price >= ma else "低于"
    diff_pct = round((price - ma) / ma * 100, 2)
    line = f"{MA_WINDOW}日均线: {_yen(ma)}円 (现价{rel}均线{abs(diff_pct)}%)"
    if price >= ma:
        line = f"🟢{line}"
    return line


def _volume_line(volume, avg_volume):
    """成交量健康度：跟20日均量比，放量/缩量/正常"""
    if not volume:
        return "成交量: 不适用(指数无成交量数据)"
    base = f"成交量: {volume:,} 股"
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


def _stock_block(stock, data, with_news):
    """组装单只股票在合并推送里的一段文字"""
    pct = data["pct"]
    direction = "涨" if pct > 0 else "跌"

    try:
        stats = get_history_stats(stock["code"])
        week52_line = f"52周: {_yen(stats['low'])}円({stats['low_date']}) ~ {_yen(stats['high'])}円({stats['high_date']})"
        ma_line = _ma_line(data["price"], stats["ma"])
        volume_line = _volume_line(data["volume"], stats["avg_volume"])
    except Exception:
        week52_line = f"52周: {_yen(data['week52_low'])}~{_yen(data['week52_high'])} 円"
        ma_line = f"{MA_WINDOW}日均线: 数据不足"
        volume_line = f"成交量: {data['volume']:,} 股"

    try:
        intraday = get_intraday_extremes(stock["code"])
        day_range_line = f"当日振幅: {_yen(intraday['low'])}円({intraday['low_time']}) ~ {_yen(intraday['high'])}円({intraday['high_time']})"
    except Exception:
        day_range_line = f"当日振幅: {_yen(data['day_low'])}~{_yen(data['day_high'])} 円"

    # 股票名标色点：涨=🔴红，跌=🟢绿(中式行情配色)，平盘不标(Server酱不渲染HTML,用emoji)
    if pct > 0:
        name_html = f'🔴{stock["name"]}'
    elif pct < 0:
        name_html = f'🟢{stock["name"]}'
    else:
        name_html = stock["name"]

    if data.get("trend_tag"):  # 收盘播报: 强势/弱势/横盘 标在名字后
        name_html += f" 【{data['trend_tag']}】"

    gap_line = f"{data['gap_note']}\n\n" if data.get("gap_note") else ""
    rel_line = f"{data['rel_note']}\n\n" if data.get("rel_note") else ""
    sig_parts = []
    if data.get("tech_stars"):
        n, label, basis = data["tech_stars"]
        sig_parts.append(f"{'★' * n}{'☆' * (5 - n)}{label}({basis})")
    if data.get("tech_signals"):
        sig_parts.extend(data["tech_signals"])
    sig_line = f"信号: {' | '.join(sig_parts)}\n\n" if sig_parts else ""
    block = (
        f"### {name_html}\n\n"
        f"{direction}{abs(pct)}% | 现价 **{_yen(data['price'])}** 円@{_quote_time_disp(data.get('quote_time'))} (昨收 {_yen(data['prev_close'])})\n\n"
        f"{gap_line}"
        f"{rel_line}"
        f"{sig_line}"
        f"{volume_line}\n\n"
        f"{day_range_line}\n\n"
        f"{ma_line}\n\n"
        f"{week52_line}"
    )
    if with_news:
        try:
            news = get_news(stock["news_query"])
        except Exception:
            news = []
        if news:
            analyzed = None
            try:
                analyzed = analyze_news_llm(stock["name"], news)
            except Exception as e:
                print(f"[{stock['name']}] LLM新闻分析失败,回退原标题: {e}")
            if analyzed is not None:
                if analyzed:  # 空列表=全部低相关,整段丢弃(降噪)
                    mark = {"利好": "🔴利好", "利空": "🟢利空", "中性": "⚪中性"}
                    block += "\n\n消息面: " + "；".join(
                        f"{mark.get(j, '⚪' + j)} [{s}]({lk})" for s, j, r, lk in analyzed
                    )
            else:
                block += "\n\n相关新闻: " + "；".join(f"[{title}]({link})" for title, link in news)
    return block


def _profit_html(profit, pct):
    mark = "🔴" if profit >= 0 else "🟢"
    sign = "+" if profit >= 0 else ""
    return f"{mark}{sign}{_yen(profit)}円 ({sign}{pct}%)"


def _margin_status(tategyoku_cost, eval_loss):
    """信用维持率测算+追证预警。tategyoku_cost=建玉代金,eval_loss=当前评价损(亏为正)。
    维持率=(受入保证金-评价损)/建玉代金;算出距追证线20%还能让建玉再跌多少%"""
    cur_val = tategyoku_cost - eval_loss
    mr = (MARGIN_DEPOSIT - eval_loss) / tategyoku_cost * 100
    # 触及追证线20%时的评价损 → 建玉还能再跌的比例
    max_loss_call = MARGIN_DEPOSIT - MARGIN_CALL_LINE / 100 * tategyoku_cost
    drop_to_call = (max_loss_call - eval_loss) / cur_val * 100 if cur_val > 0 else 0
    max_loss_force = MARGIN_DEPOSIT - MARGIN_FORCE_LINE / 100 * tategyoku_cost
    drop_to_force = (max_loss_force - eval_loss) / cur_val * 100 if cur_val > 0 else 0
    flag = "🟢" if mr > MARGIN_WARN_LINE else "🔴"
    head = f"{flag}信用维持率(估) **{mr:.1f}%**"
    if mr <= MARGIN_WARN_LINE:
        head += f" ⚠已≤预警线{MARGIN_WARN_LINE}%"
    return (
        f"{head}\n\n"
        f"距追证线{MARGIN_CALL_LINE}%: 建玉再跌{drop_to_call:.1f}% | "
        f"距强平线{MARGIN_FORCE_LINE}%: 再跌{drop_to_force:.1f}%\n\n"
        f"(估算假设受入保证金{_yen(MARGIN_DEPOSIT)}円不变;实际含现物代用担保会同步缩水,真跌时维持率降更快)"
    )


def position_report():
    """持仓盈亏报告：现物/信用分开两个分区(各自小计+按盈亏%降序)，最后总计；盈利红/亏损绿"""
    groups = {"现物": [], "信用买": []}
    totals = {"现物": [0, 0], "信用买": [0, 0]}  # [成本, 市值]
    for p in POSITIONS:
        try:
            price = get_price(p["code"])["price"]
        except Exception:
            continue
        cost_amt = p["cost"] * p["qty"]
        value = price * p["qty"]
        profit = value - cost_amt
        profit_pct = round(profit / cost_amt * 100, 2)
        groups[p["kind"]].append((profit_pct, profit, price, p))
        totals[p["kind"]][0] += cost_amt
        totals[p["kind"]][1] += value

    sections = []
    grand_cost = 0
    grand_value = 0
    for kind, label in (("现物", "现物持仓"), ("信用买", "信用持仓")):
        rows = groups[kind]
        if not rows:
            continue
        rows.sort(key=lambda r: -r[0])
        cost, value = totals[kind]
        profit = value - cost
        pct = round(profit / cost * 100, 2)
        grand_cost += cost
        grand_value += value
        lines = [
            f'{p["name"]} {p["qty"]}股 {_yen(p["cost"])}→{_yen(price)}円 {_profit_html(pf, ppct)}'
            for ppct, pf, price, p in rows
        ]
        block = (
            f"## {label}({len(rows)}只)\n\n"
            f"**小计: {_profit_html(profit, pct)}** (市值{_yen(value)}円 / 成本{_yen(cost)}円)\n\n"
            + "\n\n".join(lines)
        )
        if kind == "信用买":  # 信用仓附维持率测算+追证预警(传建玉代金和当前评价损)
            block += "\n\n" + _margin_status(cost, cost - value)
        sections.append(block)

    if not sections:
        return ""
    grand_profit = grand_value - grand_cost
    grand_pct = round(grand_profit / grand_cost * 100, 2)
    sections.append(f"**总计: {_profit_html(grand_profit, grand_pct)}** (市值{_yen(grand_value)}円 / 成本{_yen(grand_cost)}円)")
    return "\n\n---\n\n".join(sections)


def send_batch(reason, items):
    """把同一轮触发的多只股票合并成一条微信推送；reason: 异动/开盘/收盘
    排序：日经指数固定第一，其余按涨跌幅从大到小；开盘/收盘推送末尾附持仓盈亏报告；
    收盘时每股附技术信号行(RSI超买超卖/均线缠绕/异常放量/缺口回补,无信号不显示)"""
    if not items:
        return
    items = sorted(items, key=lambda pair: (0 if pair[0]["code"] == "^N225" else 1, -pair[1]["pct"]))
    if reason == "收盘":
        for stock, data in items:
            try:
                tech = tech_check_stock(stock["code"])
                if tech:
                    data["trend_tag"] = tech["trend"]
                    if tech["signals"]:
                        data["tech_signals"] = tech["signals"]
                    if tech["stars"]:
                        data["tech_stars"] = tech["stars"]
            except Exception as e:
                print(f"[{stock['name']}] 技术信号检测失败: {e}")
    title = f"日股{reason}播报({len(items)}只)"
    desp = "\n\n---\n\n".join(
        _stock_block(stock, data, with_news=(reason == "异动")) for stock, data in items
    )
    if reason in ("开盘", "收盘"):
        try:
            report = position_report()
        except Exception as e:
            report = ""
            print(f"持仓报告生成失败(不影响推送): {e}")
        if report:
            desp += "\n\n---\n\n" + report
    desp += f"\n\n推送生成: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')} (JST)  ※各股行情时刻见现价旁@标注"
    _send_text(title, desp, mail=(reason in ("开盘", "收盘")))  # 异动只发微信


def _md_to_html(desp):
    """推送正文(markdown风格)转邮件HTML：加粗/标题/链接/分隔线/换行；涨🔴跌🟢在邮件里补真颜色"""
    html = desp
    html = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', html)
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.M)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.M)
    html = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", html)
    html = html.replace("\n\n---\n\n", "<hr>")
    # emoji色点后的文字补真实颜色(邮件支持HTML,微信不支持)
    html = re.sub(r"🔴([^\s|<,，(]+)", r'🔴<span style="color:#e03131">\1</span>', html)
    html = re.sub(r"🟢([^\s|<,，(]+)", r'🟢<span style="color:#2f9e44">\1</span>', html)
    html = html.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return f'<div style="font-family:sans-serif;line-height:1.6">{html}</div>'


def send_mail(title, desp):
    """通过Gmail SMTP发邮件(HTML)；未配置应用密码则静默跳过"""
    if not GMAIL_APP_PASSWORD:
        return
    msg = MIMEText(_md_to_html(desp), "html", "utf-8")
    msg["Subject"] = Header(title, "utf-8")
    msg["From"] = GMAIL_USER
    msg["To"] = MAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, [MAIL_TO], msg.as_string())
    print(f"[邮件成功] {title}")


def _send_text(title, desp, mail=False):
    """统一推送出口：微信(Server酱)所有类型都发；邮件仅mail=True的类型(开盘/收盘/潜力股提示)。
    单渠道失败不影响另一渠道"""
    try:
        url = f"https://sctapi.ftqq.com/{SENDKEY}.send"
        requests.post(url, data={"title": title, "desp": desp}, timeout=15)
        print(f"[推送成功] {title}")
    except Exception as e:
        print(f"[微信推送失败] {title}: {e}")
    if not mail:
        return
    try:
        send_mail(title, desp)
    except Exception as e:
        print(f"[邮件推送失败] {title}: {e}")


_yahoo_sess = None


def _yahoo_session():
    """带crumb的Yahoo会话(quoteSummary日历接口需要)，进程内缓存"""
    global _yahoo_sess
    if _yahoo_sess is None:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        s.get("https://fc.yahoo.com", timeout=15)
        crumb = s.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15).text
        _yahoo_sess = (s, crumb)
    return _yahoo_sess


def get_earnings_date(code):
    """下次财报(决算发表)预定日；返回(日期字符串, 是否预估)，查不到返回(None, None)"""
    s, crumb = _yahoo_session()
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{code}"
    r = s.get(url, params={"modules": "calendarEvents", "crumb": crumb}, timeout=15)
    result = r.json().get("quoteSummary", {}).get("result")
    if not result or not result[0]:  # ETF等无财报概念的品种
        return None, None
    earnings = (result[0].get("calendarEvents") or {}).get("earnings") or {}
    dates = earnings.get("earningsDate") or []
    if not dates:
        return None, None
    d = datetime.fromtimestamp(dates[0]["raw"], JST).strftime("%Y-%m-%d")
    return d, bool(earnings.get("isEarningsDateEstimate"))


def _bdays_until(today_str, target_str):
    """从明天起数到target当天(含)的交易日数(剔除周末+日本节假日+年末年始)；target=今天返回0，已过去返回-1"""
    t0 = datetime.strptime(today_str, "%Y-%m-%d").date()
    t1 = datetime.strptime(target_str, "%Y-%m-%d").date()
    if t1 < t0:
        return -1
    n = 0
    d = t0
    while d < t1:
        d += timedelta(days=1)
        if not is_market_closed_day(d):
            n += 1
    return n


def check_earnings_reminders(today):
    """每天一次：刷新各监控股的财报预定日；距发布≤N个工作日且未提醒过 → 合并推一条预告"""
    g = state["_global"]
    if g.get("earnings_check_date") == today:
        return
    g["earnings_check_date"] = today
    lines = []
    for stock in STOCKS:
        if stock["code"].startswith("^"):
            continue
        info = g.setdefault("earnings", {}).setdefault(stock["code"], {})
        try:
            d, est = get_earnings_date(stock["code"])
        except Exception as e:
            print(f"[{stock['name']}] 财报日获取失败: {e}")
            continue
        if not d:
            continue
        info["date"] = d
        info["estimate"] = est
        bd = _bdays_until(today, d)
        if 0 <= bd <= EARNINGS_REMIND_BDAYS and info.get("reminded_for") != d:
            week = "一二三四五六日"[datetime.strptime(d, "%Y-%m-%d").weekday()]
            tag = " (日期为预估,可能变动)" if est else ""
            when = "今天" if bd == 0 else f"还有{bd}个工作日"
            lines.append(f"🗓️ {stock['name']}: {d}(周{week}) 发布财报，{when}{tag}")
            info["reminded_for"] = d
    if lines:
        _send_text(f"财报预告({len(lines)}家)", "\n\n".join(lines), mail=True)


def get_consensus(code):
    """Yahoo分析师一致预期(earningsTrend)：按期末日期索引的营收/EPS平均预期"""
    s, crumb = _yahoo_session()
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{code}"
    r = s.get(url, params={"modules": "earningsTrend", "crumb": crumb}, timeout=15)
    result = r.json().get("quoteSummary", {}).get("result")
    if not result or not result[0]:
        return {}
    out = {}
    for t in (result[0].get("earningsTrend") or {}).get("trend", []):
        end = t.get("endDate")
        rev = ((t.get("revenueEstimate") or {}).get("avg") or {}).get("raw")
        eps = ((t.get("earningsEstimate") or {}).get("avg") or {}).get("raw")
        n = ((t.get("earningsEstimate") or {}).get("numberOfAnalysts") or {}).get("raw")
        if end and (rev or eps):
            out[end] = {"revenue": rev, "eps": eps, "analysts": n}
    return out


def _tanshin_period_end(title):
    """从短信标题解析报告期末(YYYY-MM)和季度号(通期=None)。如'2027年3月期 第1四半期'→('2026-06',1)"""
    m = re.search(r"(\d{4})年(\d{1,2})月期(?:.*?第(\d)四半期)?", title)
    if not m:
        return None, None
    fy_y, fy_m, q = int(m.group(1)), int(m.group(2)), m.group(3)
    if q:
        q = int(q)
        y, mth = fy_y, fy_m - (4 - q) * 3
        while mth <= 0:
            mth += 12
            y -= 1
        return f"{y:04d}-{mth:02d}", q
    return f"{fy_y:04d}-{fy_m:02d}", None


def _vs_expectation(diff_pct):
    mark = "🔴超预期" if diff_pct >= 0 else "🟢低于预期"
    return f"{mark}{'+' if diff_pct >= 0 else ''}{diff_pct:.1f}%"


def consensus_compare(stock_code, title, parsed):
    """实绩vs市场预期对比行。仅第1四半期/通期短信可直接对比(Q2/Q3短信为累计值)；无预期数据返回None"""
    period_end, q = _tanshin_period_end(title)
    if not period_end or q in (2, 3):
        return None
    actual = parsed.get("actual", {})
    rev_act = actual.get("营收", (None, None))[0]        # 百万円
    eps_act = parsed.get("eps")                          # 円
    if rev_act is None and eps_act is None:
        return None
    cons = get_consensus(stock_code)
    match = next((v for k, v in cons.items() if str(k).startswith(period_end)), None)
    if not match:
        return None
    parts = []
    if rev_act is not None and match.get("revenue"):
        est_m = match["revenue"] / 1e6  # 円→百万円
        diff = (rev_act - est_m) / est_m * 100
        parts.append(f"营收 {_vs_expectation(diff)} (预期{_fmt_million(est_m)})")
    if eps_act is not None and match.get("eps"):
        diff = (eps_act - match["eps"]) / abs(match["eps"]) * 100
        parts.append(f"EPS {_vs_expectation(diff)} (实际{eps_act}円/预期{match['eps']:.1f}円)")
    if not parts:
        return None
    n = match.get("analysts")
    return " | ".join(parts) + (f" [分析师{n}人]" if n else "")


# 决算短信Summary XBRL的指标名族(JGAAP+IFRS)；"ChangeIn"+名 = 同比%
_TANSHIN_METRICS = [
    ("营收", ("NetSales", "OperatingRevenues", "TotalRevenues", "Sales", "Revenue",
              "NetSalesIFRS", "SalesIFRS", "TotalRevenuesIFRS", "RevenueIFRS", "OperatingRevenuesIFRS")),
    ("营业利益", ("OperatingIncome", "OperatingProfit", "OperatingIncomeIFRS", "OperatingProfitIFRS")),
    ("经常利益", ("OrdinaryIncome", "OrdinaryProfit", "ProfitBeforeTax", "ProfitBeforeTaxIFRS",
                 "IncomeBeforeIncomeTaxes")),
    ("净利润", ("ProfitAttributableToOwnersOfParent", "ProfitAttributableToOwnersOfParentIFRS",
               "NetIncome", "NetIncomeIFRS", "Profit")),
]


def _ix_items(html):
    """从inline XBRL的html里抽出所有(指标名, contextRef, 数值)"""
    items = []
    for attrs, text in re.findall(r"<ix:nonfraction([^>]*)>([^<]*)</ix:nonfraction>", html, re.I):
        m_name = re.search(r'name="[^":]*:(\w+)"', attrs)
        m_ctx = re.search(r'contextRef="([^"]+)"', attrs)
        if not m_name or not m_ctx:
            continue
        raw = text.replace(",", "").replace("△", "-").strip()
        try:
            v = float(raw)
        except ValueError:
            continue
        if 'sign="-"' in attrs:
            v = -abs(v)
        items.append((m_name.group(1), m_ctx.group(1), v))
    return items


def parse_tanshin_summary(zip_bytes):
    """解析决算短信XBRL的Summary，抽实绩与通期预想(单位:百万円)。
    返回 {"actual": {指标: (值, 同比%)}, "forecast": {...}}，解析不出返回None"""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    summary_files = [n for n in z.namelist() if "Summary" in n and n.endswith(".htm")]
    if not summary_files:
        return None
    items = _ix_items(z.read(summary_files[0]).decode("utf-8", errors="ignore"))

    def pick(names, ctx_ok):
        for nm, ctx, v in items:
            if nm in names and ctx_ok(ctx):
                return v
        return None

    is_actual = lambda ctx: ctx.startswith("Current") and "Result" in ctx
    is_forecast = lambda ctx: "Forecast" in ctx

    out = {"actual": {}, "forecast": {}, "eps": None}
    for label, names in _TANSHIN_METRICS:
        chg_names = tuple("ChangeIn" + n for n in names) + tuple("ChangesIn" + n for n in names)
        v = pick(names, is_actual)
        if v is not None:
            out["actual"][label] = (v, pick(chg_names, is_actual))
        fv = pick(names, is_forecast)
        if fv is not None:
            out["forecast"][label] = (fv, pick(chg_names, is_forecast))
    # 实绩EPS(单位:円)，用于与分析师预期对比
    out["eps"] = pick(
        ("NetIncomePerShare", "BasicNetIncomePerShare", "BasicEarningsPerShareIFRS",
         "NetIncomePerShareIFRS", "BasicEarningsPerShare", "BasicEarningsPerShareUS"),
        is_actual,
    )
    return out if (out["actual"] or out["forecast"]) else None


def parse_revision_summary(zip_bytes):
    """解析业绩预想修正公告XBRL：修正前(PreviousMember,可能为区间)与修正后(CurrentMember_ForecastMember)。
    返回 {指标: {"new": 值, "prev": [值...]}}，解析不出返回None"""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    htms = [n for n in z.namelist() if n.endswith(".htm")]
    if not htms:
        return None
    items = _ix_items(z.read(htms[0]).decode("utf-8", errors="ignore"))
    out = {}
    for label, names in _TANSHIN_METRICS:
        new = next((v for nm, ctx, v in items if nm in names and "CurrentMember_ForecastMember" in ctx), None)
        prevs = [v for nm, ctx, v in items if nm in names and "PreviousMember" in ctx]
        if new is not None or prevs:
            out[label] = {"new": new, "prev": prevs}
    return out or None


def _fmt_revision(rev):
    """修正公告的文案：每个指标显示修正后值+上调/下调幅度+修正前值"""
    lines = []
    for label, d in rev.items():
        new, prevs = d.get("new"), d.get("prev")
        if new is None:
            continue
        s = f"{label} {_fmt_million(new)}"
        if prevs:
            mid = sum(prevs) / len(prevs)
            if mid:
                chg = (new - mid) / abs(mid) * 100
                arrow = "🔴上调" if chg >= 0 else "🟢下调"
                prev_txt = "~".join(_fmt_million(p) for p in sorted(prevs))
                s += f" ({arrow}{chg:+.1f}%, 修正前{prev_txt})"
        lines.append(s)
    return " | ".join(lines)


def _fmt_million(v):
    """百万円→亿円显示"""
    return f"{v / 100:,.1f}亿円"


def _fmt_metrics(metrics, chg_label="同比"):
    """chg_label: 实绩用'同比'(vs去年同期)，通期预想用'较上财年'(公司指引vs上一财年实绩)"""
    parts = []
    for label, (v, chg) in metrics.items():
        s = f"{label} {_fmt_million(v)}"
        if chg is not None:
            s += f"({chg_label}{'+' if chg >= 0 else ''}{chg}%)"
        parts.append(s)
    return " | ".join(parts)


def _tanshin_verdict(actual):
    """简单结论：增收/减收 + 增益/减益(按营收和营业利益同比,缺营益用净利)"""
    schg = actual.get("营收", (None, None))[1]
    pchg = actual.get("营业利益", (None, None))[1] or actual.get("净利润", (None, None))[1]
    if schg is None or pchg is None:
        return None
    return ("增收" if schg >= 0 else "减收") + ("增益" if pchg >= 0 else "减益")


def check_tdnet(today):
    """每轮检查TDnet披露流：监控股的决算短信/业绩预想修正发布 → 下载XBRL解析关键数字并推送"""
    g = state["_global"]
    seen = g.setdefault("tanshin_seen", [])
    codes4 = {s["code"].split(".")[0] for s in STOCKS if not s["code"].startswith("^")}

    # 镜像feed不稳定(实测偶发超时)：重试2次，仍失败则本轮跳过(每5分钟有下一轮)
    r = None
    for attempt in range(2):
        try:
            r = requests.get(TDNET_RECENT_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            break
        except requests.RequestException as e:
            if attempt == 1:
                raise
            print(f"TDnet清单第{attempt + 1}次超时,重试: {e}")
    for it in r.json().get("items", []):
        td = it.get("Tdnet", it) or it  # 镜像2026-07-07起去掉了Tdnet包装层,兼容新旧两种结构
        tid = td.get("id")
        title = td.get("title", "")
        code4 = str(td.get("company_code", ""))[:4]
        if code4 not in codes4 or tid in seen:
            continue
        is_tanshin = "決算短信" in title and "訂正" not in title
        is_revision = ("業績予想" in title or "配当予想" in title) and "修正" in title and "訂正" not in title
        if not (is_tanshin or is_revision):
            continue
        if not str(td.get("pubdate", "")).startswith(today):
            continue  # 只处理当天发布的，防止首次运行回灌旧披露
        seen.append(tid)
        del seen[:-200]

        stock = next(s for s in STOCKS if s["code"].split(".")[0] == code4)
        zip_bytes = None
        try:
            if td.get("url_xbrl"):
                # 镜像的rd.php只是跳转,直连TDnet官方源更稳(镜像下载实测会超时)
                xbrl_url = td["url_xbrl"].split("rd.php?", 1)[-1]
                zr = requests.get(xbrl_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                zip_bytes = zr.content
        except Exception as e:
            print(f"[{stock['name']}] XBRL下载失败: {e}")

        desp = f"**{stock['name']}**\n\n{title}\n\n"

        if is_tanshin:
            parsed = None
            try:
                if zip_bytes:
                    parsed = parse_tanshin_summary(zip_bytes)
            except Exception as e:
                print(f"[{stock['name']}] 财报解析失败: {e}")
            if parsed:
                if parsed["actual"]:
                    desp += f"实绩: {_fmt_metrics(parsed['actual'], '同比')}\n\n"
                    verdict = _tanshin_verdict(parsed["actual"])
                    if verdict:
                        desp += f"结论: **{verdict}**\n\n"
                    try:
                        vs = consensus_compare(stock["code"], title, parsed)
                    except Exception as e:
                        vs = None
                        print(f"[{stock['name']}] 预期对比失败: {e}")
                    if vs:
                        desp += f"对比市场预期: {vs}\n\n"
                if parsed["forecast"]:
                    desp += f"通期预想(公司指引): {_fmt_metrics(parsed['forecast'], '较上财年')}\n\n"
            else:
                if not td.get("url_xbrl"):
                    desp += "(该公告未附带XBRL数据文件——美国会计基准短信等属正常情况，无法自动解析，请看原文PDF)\n\n"
                else:
                    desp += "(关键数字自动解析失败，请看原文PDF)\n\n"
            try:
                quality = analyze_tanshin_quality(zip_bytes, stock["name"])
                if quality:
                    desp += quality + "\n\n"
            except Exception as e:
                print(f"[{stock['name']}] 财报科目预筛失败(不影响推送): {e}")
            push_title = f"📊 财报发布: {stock['name']}"
        else:
            rev = None
            try:
                if zip_bytes:
                    rev = parse_revision_summary(zip_bytes)
            except Exception as e:
                print(f"[{stock['name']}] 修正公告解析失败: {e}")
            if rev:
                desp += f"修正后通期预想: {_fmt_revision(rev)}\n\n"
            else:
                desp += "(数值自动解析失败，请看原文PDF)\n\n"
            push_title = f"⚠️ 业绩预想修正: {stock['name']}"

        if td.get("document_url"):
            desp += f"[公告PDF原文]({td['document_url']})"
        _send_text(push_title, desp, mail=True)  # 财报发布/业绩修正也发邮箱


# ============ 潜力股筛选(每天收盘后扫描SCREEN_POOL) ============

def get_daily_ohlcv(code):
    """一年日线OHLCV序列(剔除空值日)，附股票名和日期"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?interval=1d&range=1y"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=15)
    result = resp.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    rows = [
        (ts, o, h, l, c, v)
        for ts, o, h, l, c, v in zip(result["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"])
        if c is not None and o is not None and h is not None and l is not None
    ]
    name = result["meta"].get("shortName") or code
    return {
        "name": name,
        "date": [datetime.fromtimestamp(r[0], JST).strftime("%Y-%m-%d") for r in rows],
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [r[5] or 0 for r in rows],
    }


def _sma(vals, n, back=0):
    """n日简单均线；back=0最新一天，back=1前一天"""
    end = len(vals) - back
    return sum(vals[end - n:end]) / n


def _rsi_last2(closes, n=14):
    """Wilder RSI(14)的最近两个值 (昨值, 今值)"""
    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_g = sum(gains[:n]) / n
    avg_l = sum(losses[:n]) / n
    rsis = []
    for i in range(n, len(deltas)):
        avg_g = (avg_g * (n - 1) + gains[i]) / n
        avg_l = (avg_l * (n - 1) + losses[i]) / n
        rsis.append(100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l))
    return (rsis[-2], rsis[-1]) if len(rsis) >= 2 else (None, None)


GOLDEN_CROSS_VOL_RATIO = 1.2   # 金叉当日成交量须≥该倍数×20日均量才算有效(缩量金叉不报)
CONSECUTIVE_MARK_DAYS = 5      # 连续上榜达到该天数→潜力股推送里重点标记⭐
GAP_ALERT_PCT = 1.5            # 开盘跳空幅度≥该%才在开盘播报里标注
TECH_VOL_ANOMALY = 2.0         # 成交量≥该倍数×20日均量算异常放量(技术日报)
RSI_OVERBOUGHT = 70            # RSI超买线
RSI_OVERSOLD = 30              # RSI超卖线
MA_TANGLE_PCT = 2.0            # 三条均线极差/现价<该% 判定均线缠绕(粘合待变盘)
TREND_FLAT_CHG20 = 3.0         # 20日涨幅绝对值<该% 且贴近均线 → 判横盘
TREND_FLAT_DEV = 2.0           # 现价偏离MA20<该% 视为贴近均线


def _load_hit_history():
    """读screen_hits.csv历史 → {日期: {代码,...}}"""
    hits_csv = os.path.join(BASE_DIR, "screen_hits.csv")
    hist = {}
    if os.path.exists(hits_csv):
        with open(hits_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                hist.setdefault(row["日期"], set()).add(row["代码"])
    return hist


def _streak_days(code, today, hist):
    """含今天在内的连续上榜天数：从最近一个既往筛选日往回数，断档即停"""
    prev_dates = sorted((d for d in hist if d < today), reverse=True)
    streak = 1
    for d in prev_dates:
        if code in hist[d]:
            streak += 1
        else:
            break
    return streak


def screen_stock(code):
    """对单只股票跑全部走势条件。返回(名称, 命中列表, 数据支撑dict)；
    facts含: price/day_chg/chg20(20日涨幅%,用作趋势强度排序)/vol_ratio/ma5/ma20/ma60/rsi"""
    d = get_daily_ohlcv(code)
    c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]
    if len(c) < 65:
        return d["name"], [], None
    hits = []

    ma5, ma20, ma60 = _sma(c, 5), _sma(c, 20), _sma(c, 60)
    avg_vol20 = _sma(v, 20, 1)
    vol_ratio = v[-1] / avg_vol20 if avg_vol20 > 0 else 0
    rsi_y, rsi_t = _rsi_last2(c)
    hi_idx = max(range(len(h)), key=lambda i: h[i])   # 52周最高/最低及发生日期
    lo_idx = min(range(len(l)), key=lambda i: l[i])
    facts = {
        "price": c[-1],
        "day_chg": round((c[-1] / c[-2] - 1) * 100, 2),
        "chg20": round((c[-1] / c[-21] - 1) * 100, 1),  # 20日涨幅=趋势强度
        "vol_ratio": vol_ratio,
        "ma5": ma5, "ma20": ma20, "ma60": ma60,
        "rsi": rsi_t,
        "w52_high": h[hi_idx], "w52_high_date": d["date"][hi_idx],
        "w52_low": l[lo_idx], "w52_low_date": d["date"][lo_idx],
    }

    # 1a. 短线金叉: MA5昨日≤MA20,今日上穿；须放量确认(量≥1.2倍20日均量)
    if _sma(c, 5, 1) <= _sma(c, 20, 1) and ma5 > ma20 and vol_ratio >= GOLDEN_CROSS_VOL_RATIO:
        hits.append(f"MA5({_yen(ma5)})放量上穿MA20({_yen(ma20)})短线金叉,量{vol_ratio:.1f}倍确认")
    # 1b. 中期金叉: MA20上穿MA60,同样须放量确认
    if _sma(c, 20, 1) <= _sma(c, 60, 1) and ma20 > ma60 and vol_ratio >= GOLDEN_CROSS_VOL_RATIO:
        hits.append(f"MA20({_yen(ma20)})放量上穿MA60({_yen(ma60)})中期金叉,量{vol_ratio:.1f}倍确认")

    # 2. 放量突破60日新高: 收盘>此前60日最高收盘 且 量≥1.5倍20日均量
    prior_high = max(c[-61:-1])
    if c[-1] > prior_high and vol_ratio >= 1.5:
        hits.append(f"放量突破60日新高(收{_yen(c[-1])}>前高{_yen(prior_high)},量{vol_ratio:.1f}倍)")

    # 3. RSI超卖回升: RSI14昨日<30,今日回升≥30
    if rsi_y is not None and rsi_y < 30 <= rsi_t:
        hits.append(f"RSI超卖回升({rsi_y:.1f}→{rsi_t:.1f})")

    # 4. 回踩MA20企稳: MA20向上(高于5日前) + 今日最低触及MA20附近 + 收阳收回MA20上方
    ma20_5ago = _sma(c, 20, 5)
    if ma20 > ma20_5ago and l[-1] <= ma20 * 1.01 and c[-1] > ma20 and c[-1] > o[-1]:
        hits.append(
            f"上升趋势回踩MA20企稳(最低{_yen(l[-1])}触及MA20({_yen(ma20)}),收阳{_yen(c[-1])}收回上方,MA20五日升{(ma20 / ma20_5ago - 1) * 100:.1f}%)"
        )

    return d["name"], hits, facts


def _facts_line(f):
    """数据支撑行：现价/当日涨跌/20日涨幅/量比/三均线/RSI/52周区间(带日期)"""
    if f["rsi"] is None:
        return "数据: 不足"
    return (
        f"数据: 现价{_yen(f['price'])}({f['day_chg']:+}%) | 20日涨幅{f['chg20']:+}% | "
        f"量{f['vol_ratio']:.1f}倍于20日均量 | MA5/20/60={_yen(f['ma5'])}/{_yen(f['ma20'])}/{_yen(f['ma60'])} | "
        f"RSI {f['rsi']:.0f} | 52周: {_yen(f['w52_low'])}円({f['w52_low_date']}) ~ {_yen(f['w52_high'])}円({f['w52_high_date']})"
    )


def batch_quotes(codes):
    """批量取行情(v7 quote,一次最多~100只)：{code: {"price", "volume", "name"}}"""
    s, crumb = _yahoo_session()
    out = {}
    for i in range(0, len(codes), 80):
        chunk = codes[i:i + 80]
        r = s.get(
            "https://query1.finance.yahoo.com/v7/finance/quote",
            params={"symbols": ",".join(chunk), "crumb": crumb}, timeout=30,
        )
        for q in r.json().get("quoteResponse", {}).get("result", []):
            out[q["symbol"]] = {
                "price": q.get("regularMarketPrice") or 0,
                "volume": q.get("regularMarketVolume") or 0,
                "name": q.get("shortName") or q["symbol"],
            }
    return out


def build_screen_pool():
    """每天动态选池：各领域按当日成交额(量×价)排名取前SECTOR_TOP_N，附SCREEN_EXTRA必扫"""
    all_codes = [c for codes in SECTOR_UNIVERSE.values() for c in codes]
    quotes = batch_quotes(all_codes)
    pool = []  # [(sector, code)]
    picked = set()
    for sector, codes in SECTOR_UNIVERSE.items():
        ranked = sorted(
            (c for c in codes if c in quotes),
            key=lambda c: -(quotes[c]["volume"] * quotes[c]["price"]),
        )
        for c in ranked[:SECTOR_TOP_N]:
            if c not in picked:
                pool.append((sector, c))
                picked.add(c)
    for c in SCREEN_EXTRA:
        if c not in picked:
            pool.append(("自选", c))
            picked.add(c)
    return pool


def run_screen(today):
    """动态选池后逐只跑走势条件，命中→按领域分组(组内按20日涨幅=趋势强度降序)推送+记录screen_hits.csv"""
    pool = build_screen_pool()
    print(f"[筛选]今日入选 {len(pool)} 只")
    results = []  # (sector, name, code, hits, facts)
    for sector, code in pool:
        try:
            name, hits, facts = screen_stock(code)
        except Exception as e:
            print(f"[筛选]{code} 失败: {e}")
            continue
        if hits:
            known = next((s["name"] for s in STOCKS if s["code"] == code), None)
            if not known and code in CN_NAMES:
                known = f"{CN_NAMES[code]} ({code.split('.')[0]})"
            results.append((sector, known or f"{name} ({code})", code, hits, facts))

    if results:
        # 连续上榜天数(基于既往screen_hits.csv历史,今天写入前计算)
        hist = _load_hit_history()
        streaks = {code: _streak_days(code, today, hist) for _, _, code, _, _ in results}

        # 统一排序(推送和CSV一致): 按领域顺序, 组内按20日涨幅(趋势强度)从强到弱
        sector_order = {s: i for i, s in enumerate(list(SECTOR_UNIVERSE.keys()) + ["自选"])}
        results.sort(key=lambda r: (sector_order.get(r[0], 99), -r[4]["chg20"]))

        # 连续≥N日上榜的置顶重点区
        starred = [(name, streaks[code]) for _, name, code, _, _ in results if streaks[code] >= CONSECUTIVE_MARK_DAYS]
        sections = []
        if starred:
            starred.sort(key=lambda x: -x[1])
            sections.append(
                "⭐【重点关注·持续上榜】\n\n"
                + "\n\n".join(f"🔥**{name}** 已连续{n}个交易日出现在潜力股列表" for name, n in starred)
            )

        for sector in list(SECTOR_UNIVERSE.keys()) + ["自选"]:
            rows = [r for r in results if r[0] == sector]
            if rows:
                body_parts = []
                for _, name, code, hits, facts in rows:
                    n = streaks[code]
                    star = f"⭐🔥连续{n}日上榜 " if n >= CONSECUTIVE_MARK_DAYS else ""
                    body_parts.append(
                        f"🎯{star}**{name}** (20日{facts['chg20']:+}%)\n\n命中: {' | '.join(hits)}\n\n{_facts_line(facts)}"
                    )
                sections.append(f"【{sector}】\n\n" + "\n\n".join(body_parts))
        _send_text(
            f"潜力股提示({len(results)}只{'，⭐' + str(len(starred)) + '只持续上榜' if starred else ''})",
            "\n\n---\n\n".join(sections)
            + f"\n\n(池={len(pool)}只=各领域成交额前{SECTOR_TOP_N},组内按20日涨幅排序;⭐=连续{CONSECUTIVE_MARK_DAYS}日以上上榜;条件:放量金叉(量≥{GOLDEN_CROSS_VOL_RATIO}倍)/放量破60日新高/RSI超卖回升/回踩MA20;仅走势信号,不构成买入建议)",
            mail=True,
        )
        hits_csv = os.path.join(BASE_DIR, "screen_hits.csv")
        is_new = not os.path.exists(hits_csv)
        with open(hits_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["日期", "领域", "代码", "名称", "命中条件", "现价", "当日%", "20日涨幅%", "量比", "RSI",
                            "52周低(日期)", "52周高(日期)", "连续上榜天数"])
            for sector, name, code, hits, facts in results:
                w.writerow([
                    today, sector, code, name, "；".join(hits),
                    facts["price"], facts["day_chg"], facts["chg20"],
                    round(facts["vol_ratio"], 2), round(facts["rsi"], 1) if facts["rsi"] is not None else "",
                    f"{facts['w52_low']}({facts['w52_low_date']})", f"{facts['w52_high']}({facts['w52_high_date']})",
                    streaks[code],
                ])
    else:
        print(f"[筛选]{len(pool)}只均未命中")


# ============ 技术面信号(缺口/RSI超买超卖/均线缠绕/异常放量) ============

def detect_gap(code):
    """开盘跳空检测：今开vs昨收≥GAP_ALERT_PCT%才返回描述行；越过昨日最高/最低标'真缺口'。无缺口返回None"""
    d = get_daily_ohlcv(code)
    if len(d["close"]) < 2:
        return None
    today_open = d["open"][-1]
    prev_close, prev_high, prev_low = d["close"][-2], d["high"][-2], d["low"][-2]
    gap_pct = (today_open - prev_close) / prev_close * 100
    if abs(gap_pct) < GAP_ALERT_PCT:
        return None
    if gap_pct > 0:
        real = " [真缺口:开盘越过昨日最高]" if today_open > prev_high else ""
        return f"⚡向上跳空+{gap_pct:.1f}% (开{_yen(today_open)}/昨收{_yen(prev_close)}){real}"
    real = " [真缺口:开盘低于昨日最低]" if today_open < prev_low else ""
    return f"⚡向下跳空{gap_pct:.1f}% (开{_yen(today_open)}/昨收{_yen(prev_close)}){real}"


def _ma_pattern(ma5, ma20, ma60, price):
    """均线形态：多头/空头排列 或 缠绕(三线极差<现价的MA_TANGLE_PCT%,粘合待变盘)"""
    spread = (max(ma5, ma20, ma60) - min(ma5, ma20, ma60)) / price * 100
    if spread < MA_TANGLE_PCT:
        return "缠绕", f"🌀均线缠绕(MA5/20/60极差仅{spread:.1f}%,粘合待变盘)"
    if ma5 > ma20 > ma60:
        return "多头", None
    if ma5 < ma20 < ma60:
        return "空头", None
    return "交错", None


def _trend_class(price, ma20, ma20_5ago, chg20):
    """强势/弱势/横盘三分类：现价vsMA20位置 + MA20方向 + 20日涨幅"""
    dev = (price - ma20) / ma20 * 100
    if abs(chg20) < TREND_FLAT_CHG20 and abs(dev) < TREND_FLAT_DEV:
        return "横盘"
    if price >= ma20 and ma20 >= ma20_5ago:
        return "强势"
    if price <= ma20 and ma20 <= ma20_5ago:
        return "弱势"
    return "横盘"  # 价格与均线方向矛盾的过渡期，归为横盘/整理


def _signal_stars(f):
    """按用户定义的星级规则给当日信号组合打分。f=标志位dict。返回(星数,等级名,依据)或None"""
    # ★★★★☆ 重点关注
    basis = []
    if f["bull"] and f["vol_up"]:
        basis.append("多头排列+放量上涨")
    if f["gap_up_open"]:
        basis.append("向上跳空未回补")
    if f["tangle_break"]:
        basis.append("缠绕后放量向上突破")
    if basis:
        return 4, "重点关注", "+".join(basis)
    # ★★★☆☆ 需要确认
    if f["tangle"]:
        basis.append("均线缠绕")
    if f["vol_up"] and not f["bull"]:
        basis.append("单独放量")
    if f["gap_up_filled"]:
        basis.append("向上跳空已回补")
    if basis:
        return 3, "需要确认", "+".join(basis)
    # ★★☆☆☆ 风险提示
    if f["rsi_ob"]:
        basis.append("RSI超买,不单独作为卖出依据")
    if f["rsi_os"]:
        basis.append("RSI超卖,不单独作为买入依据")
    if basis:
        return 2, "风险提示", "+".join(basis)
    # ★☆☆☆☆ 偏空信号
    if f["vol_down"]:
        basis.append("放量下跌")
    if f["gap_down_open"]:
        basis.append("向下跳空未回补")
    if f["bear"]:
        basis.append("空头排列")
    if basis:
        return 1, "偏空信号", "+".join(basis)
    return None


def tech_check_stock(code):
    """单只股票的技术面体检。返回dict: signals信号文本/facts数据行/trend强弱分类/stars星级(可None)"""
    d = get_daily_ohlcv(code)
    c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]
    if len(c) < 65:
        return None
    signals = []
    flags = {k: False for k in ("bull", "bear", "tangle", "tangle_break", "vol_up", "vol_down",
                                "gap_up_open", "gap_up_filled", "gap_down_open", "rsi_ob", "rsi_os")}

    ma5, ma20, ma60 = _sma(c, 5), _sma(c, 20), _sma(c, 60)
    pattern, tangle_note = _ma_pattern(ma5, ma20, ma60, c[-1])
    flags["bull"] = pattern == "多头"
    flags["bear"] = pattern == "空头"
    flags["tangle"] = pattern == "缠绕"
    if tangle_note:
        signals.append(tangle_note)
    if flags["bear"]:
        signals.append("📉空头排列(MA5<MA20<MA60)")

    avg_vol20 = _sma(v, 20, 1)
    vol_ratio = v[-1] / avg_vol20 if avg_vol20 > 0 else 0
    if vol_ratio >= TECH_VOL_ANOMALY:
        up = c[-1] >= c[-2]
        flags["vol_up"], flags["vol_down"] = up, not up
        signals.append(f"🔥异常放量({vol_ratio:.1f}倍于20日均量,当日收{'涨' if up else '跌'})")

    # 缠绕后放量向上突破: 5日前三线仍粘合(极差<阈值) + 今收突破三线上方 + 放量≥1.5倍
    ma5_5, ma20_5, ma60_5 = _sma(c, 5, 5), _sma(c, 20, 5), _sma(c, 60, 5)
    was_tangled = (max(ma5_5, ma20_5, ma60_5) - min(ma5_5, ma20_5, ma60_5)) / c[-6] * 100 < MA_TANGLE_PCT
    if was_tangled and c[-1] > max(ma5, ma20, ma60) and vol_ratio >= 1.5:
        flags["tangle_break"] = True
        signals.append(f"🚀均线缠绕后放量向上突破(收{_yen(c[-1])}站上三线,量{vol_ratio:.1f}倍)")

    rsi_y, rsi_t = _rsi_last2(c)
    if rsi_t is not None:
        if rsi_t >= RSI_OVERBOUGHT:
            flags["rsi_ob"] = True
            signals.append(f"⚠RSI超买({rsi_t:.0f}≥{RSI_OVERBOUGHT},短期过热注意回调)")
        elif rsi_t <= RSI_OVERSOLD:
            flags["rsi_os"] = True
            signals.append(f"❄RSI超卖({rsi_t:.0f}≤{RSI_OVERSOLD},关注止跌反弹)")

    # 当日缺口及回补状态(收盘视角)
    gap_pct = (o[-1] - c[-2]) / c[-2] * 100
    if abs(gap_pct) >= GAP_ALERT_PCT:
        if gap_pct > 0:
            filled = l[-1] <= c[-2]
            flags["gap_up_open"], flags["gap_up_filled"] = not filled, filled
            signals.append(f"⚡向上跳空+{gap_pct:.1f}%({'已回补' if filled else '未回补'})")
        else:
            filled = h[-1] >= c[-2]
            flags["gap_down_open"] = not filled
            signals.append(f"⚡向下跳空{gap_pct:.1f}%({'已回补' if filled else '未回补'})")

    day_chg = (c[-1] / c[-2] - 1) * 100
    chg20 = (c[-1] / c[-21] - 1) * 100
    trend = _trend_class(c[-1], ma20, _sma(c, 20, 5), chg20)
    rsi_txt = f"{rsi_t:.0f}" if rsi_t is not None else "-"
    facts = (
        f"现价{_yen(c[-1])}({day_chg:+.2f}%) | MA5/20/60={_yen(ma5)}/{_yen(ma20)}/{_yen(ma60)} {pattern}排列"
        f" | RSI {rsi_txt} | 量{vol_ratio:.1f}倍"
    )
    return {"signals": signals, "facts": facts, "trend": trend, "stars": _signal_stars(flags)}


PREMARKET_INDICES = [("标普500", "^GSPC"), ("纳斯达克", "^IXIC"), ("费城半导体SOX", "^SOX")]


def send_premarket():
    """盘前情报(JST 8点档)：隔夜美股收盘、美元/日元、日经期货暗示的开盘方向"""
    lines = []
    for name, code in PREMARKET_INDICES:
        try:
            d = get_price(code)
        except Exception as e:
            print(f"盘前情报 {name} 获取失败: {e}")
            continue
        mark = "🔴" if d["pct"] >= 0 else "🟢"
        lines.append(f'{mark}{name}: {d["price"]:,.1f} ({d["pct"]:+}%)')

    try:
        fx = get_price("JPY=X")
        mark = "🔴" if fx["pct"] >= 0 else "🟢"
        note = ""
        if fx["pct"] >= 0.15:
            note = " → 日元走弱,利好日股"
        elif fx["pct"] <= -0.15:
            note = " → 日元走强,利空日股"
        lines.append(f'{mark}美元/日元: {fx["price"]:.2f} ({fx["pct"]:+}%){note}')
    except Exception as e:
        print(f"盘前情报 汇率获取失败: {e}")

    try:
        fut = get_price("NIY=F")       # CME日经期货(夜盘)
        n225 = get_price("^N225")      # 早8点时price=昨日收盘
        gap = round((fut["price"] - n225["price"]) / n225["price"] * 100, 2)
        mark = "🔴" if gap >= 0 else "🟢"
        lines.append(
            f'{mark}日经期货(CME): {_yen(fut["price"])} vs 昨收{_yen(n225["price"])} '
            f'→ 暗示{"高开" if gap >= 0 else "低开"}约{gap:+}%'
        )
    except Exception as e:
        print(f"盘前情报 日经期货获取失败: {e}")

    if lines:
        _send_text("盘前情报", "\n\n".join(lines) + f"\n\n时间: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')} (JST)")


def _tier(pct_abs, threshold):
    """涨跌幅达到第几档：<阈值=0档；[阈值,阈值+STEP)=1档；[阈值+STEP,阈值+2*STEP)=2档...以此类推"""
    if pct_abs < threshold:
        return 0
    return 1 + int((pct_abs - threshold) // ALERT_STEP)


def check_stock(stock, now, today, t, is_weekday, trading, market_pct=None):
    """检查单只股票：记录历史+画图，返回本轮触发的事件[(reason, stock, data), ...]，不在这里推送。
    market_pct=日经当日涨跌幅,用于异动的大盘因子过滤"""
    events = []
    st = state[stock["code"]]
    # 开盘播报已按用户要求取消(2026-07-07)——跳空信息仍在收盘播报的技术信号行里
    need_close = is_weekday and t >= MARKET_CLOSE_PUSH_MIN and st["last_close_date"] != today

    if not (need_close or trading):
        return events

    data = get_price(stock["code"])
    log_price(stock, data)
    if need_close:
        # 走势图只在收盘画一次：原每5分钟重画并提交,一天46次×17张PNG曾让仓库单日膨胀35MB
        try:
            plot_price(stock)
        except Exception as e:
            print(f"[{stock['name']}] 画图失败(不影响推送): {e}")

    if need_close:
        events.append(("收盘", stock, data))
        st["last_close_date"] = today

    if trading:
        print(f"{now.strftime('%H:%M')} {stock['name']} 价格 {data['price']} 円, 涨跌 {data['pct']}%")

        if st["alert_date"] != today:  # 新的一天，涨跌分档重新计算
            st["alert_date"] = today
            st["up_tier"] = 0
            st["down_tier"] = 0

        pct = data["pct"]
        # 大盘因子过滤：个股涨跌须有一半以上超出日经(超额|pct-市场|≥阈值/2)才算真异动，
        # 否则只是被大盘搬运(日经-2%日16只集体-3%那种)——不报。指数自身和取不到市场数据时豁免
        excess = None
        if market_pct is not None and not stock["code"].startswith("^"):
            excess = pct - market_pct
        beta_ok = excess is None or abs(excess) >= stock["threshold"] * 0.5
        if excess is not None:
            data["rel_note"] = f"vs大盘: 日经今日{market_pct:+.2f}%,本股超额{excess:+.2f}%"

        if pct >= 0:
            tier = _tier(pct, stock["threshold"])
            if tier > st["up_tier"] and beta_ok:
                events.append(("异动", stock, data))
                st["up_tier"] = tier
        else:
            tier = _tier(-pct, stock["threshold"])
            if tier > st["down_tier"] and beta_ok:
                events.append(("异动", stock, data))
                st["down_tier"] = tier

    return events


def run_once():
    """完整检查一轮所有股票并按需推送，结束后把状态写盘"""
    now = datetime.now(JST)
    today = now.strftime("%Y-%m-%d")
    t = now.hour * 60 + now.minute
    # 交易日=非周末且非日本节假日/年末年始；节假日全部功能静默(不推旧数据)
    is_business_day = not is_market_closed_day(now.date())
    trading = is_trading_time()

    if not is_business_day:
        print(f"{now.strftime('%H:%M')} 今天东证休市(周末/节假日)，跳过所有检查")
        return

    market_pct = None
    if trading:
        try:
            market_pct = get_price("^N225")["pct"]
        except Exception as e:
            print(f"日经涨跌获取失败,本轮异动退化为绝对涨跌口径: {e}")

    all_events = []
    for stock in STOCKS:
        try:
            all_events.extend(check_stock(stock, now, today, t, is_business_day, trading, market_pct))
        except Exception as e:
            print(f"[{stock['name']}] 出错了(不影响继续运行): {e}")

    for reason in ("收盘", "异动"):
        items = [(s, d) for r, s, d in all_events if r == reason]
        send_batch(reason, items)

    if is_business_day:
        g = state["_global"]
        if t < MARKET_OPEN_MIN and g.get("premarket_date") != today:
            try:
                send_premarket()
                g["premarket_date"] = today
            except Exception as e:
                print(f"盘前情报失败(下轮重试): {e}")
        try:
            check_earnings_reminders(today)
        except Exception as e:
            print(f"财报日程检查失败(不影响其他功能): {e}")
        try:
            check_tdnet(today)
        except Exception as e:
            print(f"TDnet披露检查失败(不影响其他功能): {e}")
        if t >= MARKET_CLOSE_PUSH_MIN and g.get("screen_date") != today:
            try:
                run_screen(today)
                g["screen_date"] = today
            except Exception as e:
                print(f"潜力股筛选失败(下轮重试): {e}")

    if not trading:
        print(f"{now.strftime('%H:%M')} 非交易时段")

    save_state()


def main():
    names = "、".join(s["name"] for s in STOCKS)
    print(f"监控已启动: {names} | 每{CHECK_INTERVAL//60}分钟检查一次 | 开盘/收盘/异动均合并成一条推送")

    while True:
        run_once()
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_once()
    else:
        main()
