# -*- coding: utf-8 -*-
"""日股因子研究工作台 (GUI)

四个标签页：
  1. 数据    — 下载/加载面板，查看股票池状态
  2. 因子回测 — 选因子+参数+中性化+成本 → IC/分组/衰减 三联图 + 统计表
  3. 信号点位 — 选个股 → 价格图上标注因子触发点 + 每个点之后的实际收益
  4. 登记册   — 所有试验 + 多重检验校正门槛

打包: pyinstaller quant_app.spec --noconfirm
"""
from __future__ import annotations

import os
import sys
import traceback

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import costs as costs_mod
import factors as fac
import research_log
from data import build_panel, liquidity_filter, to_wide
from evaluate import (TRADING_DAYS, align, compute_ic, forward_returns, ic_decay,
                      ic_summary, quantile_returns)
from neutralize import build_sector_map, build_size_factor, neutralize
from signals import forward_outcome, signal_points

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

FACTORS = {
    "20日反转": lambda px, p: fac.reversal_20d(px, p["window"]),
    "12-1月动量": lambda px, p: fac.momentum_12_1(px, p["lookback"], p["skip"]),
    "低波动": lambda px, p: fac.volatility(px, p["window"]),
}


class Worker(QtCore.QThread):
    """后台线程：避免下载/回测卡住界面"""
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, **self.kwargs))
        except Exception:
            self.failed.emit(traceback.format_exc())


class Canvas(FigureCanvas):
    """可交互画布：滚轮缩放、拖动平移、快捷键(R重置/←→平移/上下缩放)"""

    def __init__(self, nrows=1, ncols=1, figsize=(11, 4.5), interactive: bool = False):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.axes = self.fig.subplots(nrows, ncols)
        self._home = None      # 初始视图范围,用于R键重置
        self._drag = None
        if interactive:
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.mpl_connect("scroll_event", self._on_scroll)
            self.mpl_connect("button_press_event", self._on_press)
            self.mpl_connect("motion_notify_event", self._on_motion)
            self.mpl_connect("button_release_event", self._on_release)
            self.mpl_connect("key_press_event", self._on_key)

    def clear(self, nrows=1, ncols=1):
        self.fig.clear()
        self.axes = self.fig.subplots(nrows, ncols)
        self._home = None
        return self.axes

    def _all_axes(self):
        return list(self.fig.get_axes())

    def save_home(self):
        self._home = [(ax.get_xlim(), ax.get_ylim()) for ax in self._all_axes()]

    # ---- 滚轮缩放：以光标位置为中心，X轴联动(多子图共享时间轴) ----
    def _on_scroll(self, ev):
        if ev.inaxes is None:
            return
        if self._home is None:
            self.save_home()
        scale = 0.8 if ev.button == "up" else 1.25    # 上滚放大
        x, y = ev.xdata, ev.ydata
        shift = QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier
        for ax in self._all_axes():
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
            if ax is ev.inaxes and not shift:          # Shift+滚轮=只缩X轴
                y0, y1 = ax.get_ylim()
                ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        self.draw_idle()

    # ---- 左键拖动平移 ----
    def _on_press(self, ev):
        if ev.button == 1 and ev.inaxes is not None:
            if self._home is None:
                self.save_home()
            self._drag = (ev.x, ev.y, [(ax, ax.get_xlim(), ax.get_ylim())
                                       for ax in self._all_axes()])

    def _on_motion(self, ev):
        if not self._drag or ev.x is None:
            return
        x0, y0, states = self._drag
        for ax, xlim, ylim in states:
            inv = ax.transData.inverted()
            p0 = inv.transform((x0, y0))
            p1 = inv.transform((ev.x, ev.y))
            dx, dy = p0[0] - p1[0], p0[1] - p1[1]
            ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            if ax is ev.inaxes:
                ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.draw_idle()

    def _on_release(self, ev):
        self._drag = None

    # ---- 快捷键 ----
    def _on_key(self, ev):
        if ev.key in ("r", "R", "home"):               # 重置视图
            if self._home:
                for ax, (xl, yl) in zip(self._all_axes(), self._home):
                    ax.set_xlim(xl); ax.set_ylim(yl)
                self.draw_idle()
            return
        if self._home is None:
            self.save_home()
        step_map = {"left": -0.1, "right": 0.1}
        zoom_map = {"up": 0.85, "down": 1.18, "+": 0.85, "-": 1.18, "=": 0.85}
        if ev.key in step_map:                          # 左右平移
            for ax in self._all_axes():
                x0, x1 = ax.get_xlim(); d = (x1 - x0) * step_map[ev.key]
                ax.set_xlim(x0 + d, x1 + d)
            self.draw_idle()
        elif ev.key in zoom_map:                        # 上下/加减缩放
            s = zoom_map[ev.key]
            for ax in self._all_axes():
                x0, x1 = ax.get_xlim(); c = (x0 + x1) / 2; w = (x1 - x0) * s / 2
                ax.set_xlim(c - w, c + w)
            self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("日股因子研究工作台")
        self.resize(1360, 880)
        self.panel = self.prices = self.mask = None
        self.sector = self.size = None
        self.last_result = None
        self.worker = None

        tabs = QtWidgets.QTabWidget()
        self.tabs = tabs
        tabs.addTab(self._tab_data(), "① 数据")
        tabs.addTab(self._tab_screener(), "② 每日领域榜 ★")
        tabs.addTab(self._tab_diagnosis(), "③ 个股趋势诊断")
        tabs.addTab(self._tab_portfolio(), "④ 我的持仓 ★")
        tabs.addTab(self._tab_backtest(), "⑤ 因子回测(研究用)")
        tabs.addTab(self._tab_signal(), "⑥ 信号点位")
        tabs.addTab(self._tab_log(), "⑦ 试验登记册")
        self.setCentralWidget(tabs)
        self.tab_diag_index = 2
        self.statusBar().showMessage("启动中 — 正在检查数据是否最新…")
        QtCore.QTimer.singleShot(300, self.auto_startup)   # 界面显示后再检查,不阻塞启动

    # ---------- 数据页 ----------
    def _tab_data(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QHBoxLayout()
        self.cb_market = QtWidgets.QComboBox()
        self.cb_market.addItems(["prime", "standard", "growth"])
        self.sp_years = QtWidgets.QSpinBox()
        self.sp_years.setRange(1, 20)
        self.sp_years.setValue(10)
        self.chk_refresh = QtWidgets.QCheckBox("强制重新下载")
        self.btn_load = QtWidgets.QPushButton("加载 / 下载面板")
        self.btn_load.clicked.connect(self.on_load)
        for lbl, wid in [("市场:", self.cb_market), ("年数:", self.sp_years)]:
            form.addWidget(QtWidgets.QLabel(lbl))
            form.addWidget(wid)
        form.addWidget(self.chk_refresh)
        form.addWidget(self.btn_load)
        form.addStretch()
        lay.addLayout(form)

        self.txt_data = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_data.setPlainText(
            "说明：\n"
            "· 数据源 Yahoo Finance(非官方接口)，股票池来自 JPX 官方上市银柄一览。\n"
            "· 价格用复权价(adjclose)，已含分红/拆股调整；停牌日留空不填充。\n"
            "· 首次下载约1分钟(1559只×10年)，之后走本地 parquet 缓存秒开。\n\n"
            "⚠生存者偏差：JPX名录是当前快照，不含已退市公司，回测收益会被系统性高估。\n"
            "  因子的相对排序影响较小，绝对收益不可当真。"
        )
        lay.addWidget(self.txt_data)
        return w

    def auto_startup(self):
        """启动时自动检查数据新鲜度：已是最新→直接读缓存；过期或缺失→自动下载"""
        from data import panel_status
        mk, yr = self.cb_market.currentText(), self.sp_years.value()
        st = panel_status(mk, yr)
        if not st["exists"]:
            msg = "首次运行，正在下载数据(约1分钟)…"
        elif not st["fresh"]:
            latest = st["latest"].date() if st["latest"] is not None else "?"
            msg = f"数据仅到 {latest}，应有 {st['expected'].date()} → 自动更新中…"
        else:
            msg = f"数据已是最新({st['latest'].date()})，直接加载…"
        self.statusBar().showMessage(msg)
        self.txt_data.appendPlainText(f"\n[启动检查] {msg}")
        self.on_load(auto=True)

    def on_load(self, auto: bool = False):
        self.btn_load.setEnabled(False)
        if not auto:
            self.statusBar().showMessage("下载/加载中…")
        mk, yr = self.cb_market.currentText(), self.sp_years.value()
        rf = self.chk_refresh.isChecked() and not auto
        self.worker = Worker(build_panel, market=mk, years=yr, refresh=rf,
                             auto_refresh=auto)
        self.worker.done.connect(self._loaded)
        self.worker.failed.connect(self._err)
        self.worker.start()

    def _loaded(self, panel):
        self.panel = panel
        self.prices = to_wide(panel, "adjclose")
        self.mask = liquidity_filter(panel, min_turnover_yen=1e8)
        try:
            self.sector = build_sector_map(level="sector17")
            self.size = build_size_factor(self.prices)
        except Exception:
            self.sector = self.size = None
        # ticker → 日文公司名 (JPX官方名录)
        try:
            from universe import load_universe
            u = load_universe((self.cb_market.currentText(),))
            self.names = dict(zip(u["ticker"], u["name"]))
        except Exception:
            self.names = {}
        labels = [self._label(t) for t in self.prices.columns]
        self.cb_ticker.clear(); self.cb_ticker.addItems(labels)
        self.cb_dx.clear(); self.cb_dx.addItems(labels)
        self._refresh_fav()
        self._load_pos_table()
        self.txt_data.appendPlainText(
            f"\n[已加载] {self.prices.shape[1]}只 × {self.prices.shape[0]}交易日 "
            f"| {self.prices.index.min().date()} ~ {self.prices.index.max().date()}"
            f"\n流动性过滤后日均可交易 {self.mask.sum(axis=1).mean():.0f}只"
            f" | 中性化数据: {'就绪' if self.sector is not None else '不可用'}")
        self.btn_load.setEnabled(True)
        self.statusBar().showMessage(
            f"数据就绪 (最新 {self.prices.index[-1].date()}) — 可到「每日领域榜」扫描")
        if self.tabs.currentIndex() == 0:   # 启动后自动跳到领域榜并扫描
            self.tabs.setCurrentIndex(1)
            QtCore.QTimer.singleShot(200, self.on_scan)

    def _err(self, tb):
        self.btn_load.setEnabled(True)
        self.btn_run.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "出错", tb[-1500:])
        self.statusBar().showMessage("出错")

    # ---------- 每日领域榜 ----------
    def _tab_screener(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        self.sp_minscore = QtWidgets.QSpinBox(); self.sp_minscore.setRange(1, 100); self.sp_minscore.setValue(80)
        self.sp_minscore.setSingleStep(5)
        self.sp_minscore.setSuffix(" 分以上(满分100)")
        self.sp_topn = QtWidgets.QSpinBox(); self.sp_topn.setRange(1, 10); self.sp_topn.setValue(3)
        self.sp_topn.setSuffix(" 只/领域")
        from screener import MODE_REVERSAL, MODE_STRONG
        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems([MODE_STRONG, MODE_REVERSAL])
        self.cb_mode.setToolTip(
            "强势=选跑赢同行的(历史检验t=-5.87,是吃亏的一边)\n"
            "反转=选跑输同行的(与验证方向一致,日股是反转市场)")
        self.sp_exc = QtWidgets.QDoubleSpinBox()
        self.sp_exc.setRange(0, 50); self.sp_exc.setValue(5.0); self.sp_exc.setSuffix(" %")
        self.sp_exc.setToolTip("同行超额门槛：强势模式=超额≥此值；反转模式=超额≤-此值")
        self.sp_zone = QtWidgets.QDoubleSpinBox()
        self.sp_zone.setRange(1, 20); self.sp_zone.setValue(5.0); self.sp_zone.setSuffix(" %")
        self.sp_zone.setToolTip("现价落在「买入参考位」±此范围内 → 整行绿色高亮")
        btn = QtWidgets.QPushButton("扫描今日")
        btn.setStyleSheet("font-weight:bold; padding:6px 18px;")
        btn.clicked.connect(self.on_scan)
        b_exp = QtWidgets.QPushButton("全部展开"); b_exp.clicked.connect(lambda: self.tree_scr.expandAll())
        b_col = QtWidgets.QPushButton("全部折叠"); b_col.clicked.connect(lambda: self.tree_scr.collapseAll())
        b_fav = QtWidgets.QPushButton("★ 收藏选中")
        b_fav.setToolTip("把选中的股票加入收藏")
        b_fav.clicked.connect(self.on_fav_from_tree)
        for wd in (QtWidgets.QLabel("筛选:"), self.sp_minscore, self.sp_topn,
                   QtWidgets.QLabel("方向:"), self.cb_mode,
                   QtWidgets.QLabel("同行超额门槛:"), self.sp_exc,
                   QtWidgets.QLabel("买入区范围:"), self.sp_zone,
                   btn, b_exp, b_col, b_fav):
            bar.addWidget(wd)
        bar.addStretch()
        self.lbl_scan = QtWidgets.QLabel("请先加载数据")
        bar.addWidget(self.lbl_scan)
        lay.addLayout(bar)

        note = QtWidgets.QLabel(
            "⚠ 这两个筛选条件，用日股10年数据回头验过，结果是坏消息，如实告诉你：\n"
            "  ① 10项判据得分高的股票 → 之后20天上涨概率 52.9%，"
            "反而低于随便哪天买入的 54.6%（这个差距纯属巧合的概率约万分之一，也就是说是真的）\n"
            "  ② 同行超额高的股票 → 之后20天反而更容易跌（纯属巧合的概率小于千万分之一，"
            "几乎可以确定）。所以「强势模式」是历史上吃亏的那一边\n"
            "  ③ 日股是「涨多了会回调」的市场，「反转模式」才和数据方向一致\n"
            "→ 本榜只告诉你「今天谁的技术面强/弱」，不是买入建议。选哪边由你，数据我摆在这里。")
        note.setWordWrap(True)
        note.setStyleSheet("background:#fff3bf; padding:7px; border-radius:4px; font-size:12px;")
        lay.addWidget(note)

        self.tree_scr = QtWidgets.QTreeWidget()
        self.tree_scr.setColumnCount(9)
        self.tree_scr.setHeaderLabels(["领域 / 股票", "公司名", "得分", "现价",
                                       "买入参考", "距买入位", "20日涨幅", "同行超额", "60日涨幅"])
        self.tree_scr.setColumnCount(9)
        self.tree_scr.setAlternatingRowColors(True)
        self.tree_scr.itemDoubleClicked.connect(self.on_tree_pick)
        self.tree_scr.itemClicked.connect(self._tree_hint)
        lay.addWidget(self.tree_scr)

        # 回调风险参考表(可折叠)
        self.grp_risk = QtWidgets.QGroupBox("📊 回调风险实测：过去20天涨跌 → 之后20天的完整表现 (点标题折叠)")
        self.grp_risk.setCheckable(True)
        self.grp_risk.setChecked(True)
        self.grp_risk.toggled.connect(lambda on: self.tbl_risk.setVisible(on))
        rl = QtWidgets.QVBoxLayout(self.grp_risk)
        self.tbl_risk = QtWidgets.QTableWidget()
        self.tbl_risk.setMaximumHeight(240)
        self._fill_risk_table()
        rl.addWidget(self.tbl_risk)
        note2 = QtWidgets.QLabel(
            "读法：胜率高≠赚得多。「涨>50%」胜率最低(46.4%)但平均收益却排第二(+2.18%)——"
            "因为它中位数是 -1.30%(多数在亏)，全靠少数暴涨拉高平均，是彩票型分布，"
            "赢时+19.7%/输时-13.0%，波动极大。\n"
            "「跌>20%」则是胜率(63.3%)、中位数(+4.34%)、平均(+5.15%)三项全优，分布健康。\n"
            "⚠但「跌>20%」这档受生存者偏差影响最大：跌完之后退市/破产的公司不在数据里，"
            "真实表现会比这个数字差。")
        note2.setWordWrap(True)
        note2.setStyleSheet("font-size:12px; color:#495057; padding:4px;")
        rl.addWidget(note2)
        lay.addWidget(self.grp_risk)
        lay.addWidget(QtWidgets.QLabel("双击任意股票 → 自动跳转到「个股趋势诊断」并显示完整数据"))
        return w

    # 回调风险参考表(Prime全市场10年264万样本实测,静态结果)
    RISK_TABLE = [
        ("跌>20%", "63.3%", "+5.15%", "+4.34%", "+13.70%", "-9.60%", "1.43", "-18.4%", "+30.8%"),
        ("跌10-20%", "57.3%", "+1.91%", "+1.62%", "+9.02%", "-7.63%", "1.18", "-15.8%", "+20.3%"),
        ("跌0-10%", "55.4%", "+1.20%", "+0.96%", "+7.05%", "-6.06%", "1.16", "-12.7%", "+15.7%"),
        ("涨0-10%", "53.9%", "+1.02%", "+0.71%", "+6.97%", "-5.93%", "1.18", "-12.4%", "+15.3%"),
        ("涨10-20%", "52.5%", "+1.17%", "+0.56%", "+8.47%", "-6.90%", "1.23", "-14.6%", "+18.6%"),
        ("涨20-30%", "51.3%", "+1.37%", "+0.39%", "+10.45%", "-8.19%", "1.28", "-17.2%", "+22.8%"),
        ("涨30-50%", "51.1%", "+1.79%", "+0.38%", "+12.68%", "-9.58%", "1.32", "-19.3%", "+27.7%"),
        ("涨>50%", "46.4%", "+2.18%", "-1.30%", "+19.66%", "-12.97%", "1.52", "-26.8%", "+43.1%"),
        ("— 全样本 —", "54.6%", "+1.26%", "+0.87%", "", "", "", "", ""),
    ]

    def _fill_risk_table(self):
        heads = ["过去20天", "上涨概率", "平均收益", "中位数", "赢时均涨", "输时均跌",
                 "盈亏比", "最差5%", "最好5%"]
        t = self.tbl_risk
        t.setColumnCount(len(heads)); t.setRowCount(len(self.RISK_TABLE))
        t.setHorizontalHeaderLabels(heads)
        t.verticalHeader().setVisible(False)
        for i, row in enumerate(self.RISK_TABLE):
            for j, v in enumerate(row):
                it = QtWidgets.QTableWidgetItem(v)
                if j > 0:
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if row[0] == "— 全样本 —":
                    f = it.font(); f.setItalic(True); it.setFont(f)
                elif j == 3 and v.startswith("-"):      # 中位数为负=多数样本在亏
                    it.setForeground(QtCore.Qt.red)
                    f = it.font(); f.setBold(True); it.setFont(f)
                elif j == 1 and float(v.rstrip("%")) < 50:
                    it.setForeground(QtCore.Qt.red)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()

    def on_scan(self):
        if self.prices is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在「数据」页加载面板")
            return
        self.lbl_scan.setText("扫描中…")
        QtWidgets.QApplication.processEvents()
        try:
            from screener import scan, scan_summary
            vol = to_wide(self.panel, "volume", scrub=False)
            res = scan(self.prices, vol, min_score=self.sp_minscore.value(),
                       top_n=self.sp_topn.value(), market=self.cb_market.currentText(),
                       mode=self.cb_mode.currentText(), min_excess=self.sp_exc.value() / 100,
                       highs=to_wide(self.panel, "high"), lows=to_wide(self.panel, "low"),
                       entry_tol=self.sp_zone.value() / 100)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "扫描失败", traceback.format_exc()[-1200:])
            self.lbl_scan.setText("扫描失败")
            return

        self.tree_scr.clear()
        for sector, rows in res.items():
            n_zone = sum(1 for r in rows if r.get("in_zone"))
            top = QtWidgets.QTreeWidgetItem([
                f"【{sector}】", f"{len(rows)}只入选" + (f"  ⭐{n_zone}只在买入区" if n_zone else ""),
                "", "", "", "", "", "", ""])
            f = top.font(0); f.setBold(True); top.setFont(0, f)
            top.setBackground(0, QtCore.Qt.lightGray)
            for r in rows:
                nm = getattr(self, "names", {}).get(r["ticker"], "")
                entry = r.get("entry")
                gap = r.get("entry_gap")
                child = QtWidgets.QTreeWidgetItem([
                    r["ticker"], nm, f"{r['score']:.0f}分", f"{r['price']:,.0f}",
                    f"{entry:,.0f}" if entry else "-",
                    f"{gap:+.1%}" if gap is not None else "-",
                    f"{r['chg20']:+.1%}", f"{r['excess20']:+.1%}", f"{r['chg60']:+.1%}"])
                child.setData(0, QtCore.Qt.UserRole, r["ticker"])
                child.setForeground(7, QtCore.Qt.red if r["excess20"] > 0 else QtCore.Qt.darkGreen)
                if r["chg20"] > 0.20:              # 涨幅过大→回调风险
                    child.setForeground(6, QtCore.Qt.darkYellow)
                if r.get("in_zone"):               # 现价在买入参考位±5%内 → 整行高亮
                    for c in range(9):
                        child.setBackground(c, QtGui.QColor("#d3f9d8"))
                    ft = child.font(0); ft.setBold(True)
                    for c in (0, 1, 4, 5):
                        child.setFont(c, ft)
                    child.setText(0, "⭐ " + r["ticker"])
                    rr = r.get("rr")
                    child.setToolTip(0, f"现价距买入参考位 {gap:+.1%}(±5%内)\n"
                                        f"买入{entry:,.0f} / 止损{r.get('stop', 0):,.0f} / "
                                        f"目标{r.get('target', 0):,.0f}"
                                        + (f" / 盈亏比{rr:.2f}" if rr == rr else ""))
                top.addChild(child)
            if not rows:
                top.addChild(QtWidgets.QTreeWidgetItem(["(无符合条件的股票)"]))
            self.tree_scr.addTopLevelItem(top)
        self.tree_scr.expandAll()
        for i in range(7):
            self.tree_scr.resizeColumnToContents(i)
        d = self.prices.index[-1].date()
        self.lbl_scan.setText(f"数据 {d} | {self.cb_mode.currentText()} "
                              f"超额门槛{self.sp_exc.value():.0f}% | {scan_summary(res)}")
        self.statusBar().showMessage(f"扫描完成: {scan_summary(res)}")

    def _tree_hint(self, item, col):
        tk = item.data(0, QtCore.Qt.UserRole)
        if tk:
            self.statusBar().showMessage(f"{self._label(tk)} — 双击查看完整诊断")

    def on_tree_pick(self, item, col):
        tk = item.data(0, QtCore.Qt.UserRole)
        if not tk:
            item.setExpanded(not item.isExpanded())   # 双击领域行=折叠/展开
            return
        self.cb_dx.setCurrentText(self._label(tk))
        self.tabs.setCurrentIndex(self.tab_diag_index)
        self.on_diagnose()

    def on_fav_from_tree(self):
        import watchlist
        items = self.tree_scr.selectedItems()
        added = []
        for it in items:
            tk = it.data(0, QtCore.Qt.UserRole)
            if tk and not watchlist.contains(tk):
                watchlist.add(tk)
                added.append(tk)
        self._refresh_fav()
        self.statusBar().showMessage(
            f"已收藏 {len(added)} 只: {', '.join(added)}" if added else "未选中股票(或已在收藏中)")

    # ---------- 我的持仓 ----------
    def _tab_portfolio(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        b_calc = QtWidgets.QPushButton("刷新盈亏 + 情景测算")
        b_calc.setStyleSheet("font-weight:bold; padding:6px 16px;")
        b_calc.clicked.connect(self.on_portfolio)
        b_add = QtWidgets.QPushButton("＋ 添加持仓"); b_add.clicked.connect(self.on_pos_add)
        b_del = QtWidgets.QPushButton("－ 删除选中"); b_del.clicked.connect(self.on_pos_del)
        b_save = QtWidgets.QPushButton("保存改动"); b_save.clicked.connect(self.on_pos_save)
        for x in (b_calc, b_add, b_del, b_save):
            bar.addWidget(x)
        bar.addStretch()
        self.lbl_pf = QtWidgets.QLabel("")
        self.lbl_pf.setStyleSheet("font-size:14px; font-weight:bold;")
        bar.addWidget(self.lbl_pf)
        lay.addLayout(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        # 上：持仓表(可编辑数量/成本)
        up = QtWidgets.QWidget(); ul = QtWidgets.QVBoxLayout(up)
        ul.addWidget(QtWidgets.QLabel(
            "持仓明细(数量/成本可直接双击修改，改完点「保存改动」；数据存 portfolio.json)"))
        self.tbl_pos = QtWidgets.QTableWidget()
        self.tbl_pos.setColumnCount(10)
        self.tbl_pos.setHorizontalHeaderLabels(
            ["代码", "公司名", "类别", "数量", "成本", "现价", "市值", "盈亏", "盈亏%", "上行/下行"])
        self.tbl_pos.itemSelectionChanged.connect(self._on_pos_select)
        ul.addWidget(self.tbl_pos)
        split.addWidget(up)
        # 下：情景测算
        dn = QtWidgets.QWidget(); dl = QtWidgets.QVBoxLayout(dn)
        dl.addWidget(QtWidgets.QLabel("情景测算(点上方任一行查看该股详情)"))
        self.txt_pf = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_pf.setStyleSheet("font-family: Consolas, monospace; font-size:12px;")
        dl.addWidget(self.txt_pf)
        split.addWidget(dn)
        split.setSizes([380, 420])
        lay.addWidget(split)
        return w

    _extra_cache: dict = {}

    def _fetch_extra(self, tickers: list[str]) -> dict:
        """抓取股票池之外的标的(ETF等)。结果缓存在内存，避免每次刷新都重下"""
        from data import _fetch_one
        import requests
        out = {}
        need = []
        for t in tickers:
            if t in self._extra_cache:
                out[t] = self._extra_cache[t]
            else:
                need.append(t)
        if need:
            s = requests.Session()
            s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            for t in need:
                df = _fetch_one(s, t, self.sp_years.value())
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                cols = {f: df[f] for f in ("open", "high", "low", "close", "adjclose", "volume")}
                self._extra_cache[t] = cols
                out[t] = cols
        return out

    def _load_pos_table(self):
        import portfolio
        rows = portfolio.load()
        t = self.tbl_pos
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate([r["ticker"], r.get("name", ""), r.get("kind", "现物"),
                                   str(r["qty"]), f"{r['cost']:g}"]):
                it = QtWidgets.QTableWidgetItem(v)
                if j in (0, 1, 2):     # 代码/名称/类别只读
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                t.setItem(i, j, it)
            for j in range(5, 10):
                t.setItem(i, j, QtWidgets.QTableWidgetItem(""))
        t.resizeColumnsToContents()
        return rows

    def on_pos_add(self):
        tk, ok = QtWidgets.QInputDialog.getText(self, "添加持仓", "股票代码(如 7011 或 7011.T):")
        if not ok or not tk.strip():
            return
        tk = self._to_ticker(tk)
        qty, ok = QtWidgets.QInputDialog.getInt(self, "添加持仓", f"{tk} 持股数量:", 100, 1, 10 ** 7)
        if not ok:
            return
        cost, ok = QtWidgets.QInputDialog.getDouble(self, "添加持仓", f"{tk} 成本单价(円):",
                                                    1000.0, 0.01, 10 ** 7, 2)
        if not ok:
            return
        kind, ok = QtWidgets.QInputDialog.getItem(self, "添加持仓", "类别:", ["现物", "信用买"], 0, False)
        if not ok:
            return
        import portfolio
        rows = portfolio.load()
        rows.append({"ticker": tk, "name": getattr(self, "names", {}).get(tk, ""),
                     "qty": qty, "cost": cost, "kind": kind})
        portfolio.save(rows)
        self._load_pos_table()
        self.statusBar().showMessage(f"已添加 {tk}")

    def on_pos_del(self):
        rows_sel = {i.row() for i in self.tbl_pos.selectedItems()}
        if not rows_sel:
            return
        import portfolio
        rows = portfolio.load()
        keep = [r for i, r in enumerate(rows) if i not in rows_sel]
        portfolio.save(keep)
        self._load_pos_table()
        self.statusBar().showMessage(f"已删除 {len(rows) - len(keep)} 条")

    def on_pos_save(self):
        import portfolio
        rows = portfolio.load()
        for i, r in enumerate(rows):
            if i >= self.tbl_pos.rowCount():
                break
            try:
                r["qty"] = int(float(self.tbl_pos.item(i, 3).text()))
                r["cost"] = float(self.tbl_pos.item(i, 4).text())
            except (ValueError, AttributeError):
                continue
        portfolio.save(rows)
        self.statusBar().showMessage("持仓已保存")
        self.on_portfolio()

    def on_portfolio(self):
        if self.prices is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在「数据」页加载面板")
            return
        import portfolio
        from levels import trade_levels
        rows = self._load_pos_table()
        if not rows:
            self.lbl_pf.setText("无持仓")
            return
        self.statusBar().showMessage("测算中…")
        QtWidgets.QApplication.processEvents()
        hi = to_wide(self.panel, "high"); lo = to_wide(self.panel, "low")
        vol = to_wide(self.panel, "volume", scrub=False)
        # 持仓里可能有不在股票池的标的(如ETF 1579不属于Prime内国株) → 按需单独抓取
        missing = [r["ticker"] for r in rows if r["ticker"] not in self.prices.columns]
        extra = {}
        if missing:
            self.statusBar().showMessage(f"补取 {len(missing)} 只池外标的数据…")
            QtWidgets.QApplication.processEvents()
            extra = self._fetch_extra(missing)

        def _series(tk, field):
            if tk in extra:
                return extra[tk][field]
            return {"adjclose": self.prices, "high": hi, "low": lo, "volume": vol}[field][tk]

        # ⚠必须与表格行一一对应：跳过的标的用None占位，否则点击行会取错数据
        self._pf_results = [None] * len(rows)
        for i, r in enumerate(rows):
            tk = r["ticker"]
            if tk not in self.prices.columns and tk not in extra:
                continue
            try:
                close = _series(tk, "adjclose").dropna()
                price = float(close.iloc[-1])
            except Exception:
                continue
            lv = None
            try:
                lv = trade_levels(_series(tk, "high").dropna(), _series(tk, "low").dropna(),
                                  close, _series(tk, "volume").dropna())
            except Exception:
                pass
            a = portfolio.analyze(r, price, lv)
            a["_levels"] = lv
            self._pf_results[i] = a
            for j, v in [(5, f"{price:,.0f}"), (6, f"{a['value']:,.0f}"),
                         (7, f"{a['pnl']:+,.0f}"), (8, f"{a['pnl_pct']:+.1f}%"),
                         (9, f"{a['ratio']:.2f}" if a.get("ratio") == a.get("ratio") else "-")]:
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if j in (7, 8):
                    it.setForeground(QtCore.Qt.red if a["pnl"] >= 0 else QtCore.Qt.darkGreen)
                self.tbl_pos.setItem(i, j, it)
        self.tbl_pos.resizeColumnsToContents()

        valid = [a for a in self._pf_results if a]
        t = portfolio.totals(valid)
        self.lbl_pf.setText(
            f"总市值 {t['value']:,.0f}円   成本 {t['cost_amt']:,.0f}円   "
            f"盈亏 {t['pnl']:+,.0f}円 ({t['pnl_pct']:+.1f}%)")
        self.lbl_pf.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{'#c92a2a' if t['pnl'] >= 0 else '#2f9e44'};")

        L = ["══ 组合层面情景测算 ══",
             f"  当前     : 市值 {t['value']:>12,.0f}円   盈亏 {t['pnl']:>+11,.0f}円 ({t['pnl_pct']:+.1f}%)"]
        for k, lbl in [("T1", "全部到T1"), ("T2", "全部到T2"), ("T3", "全部到T3"), ("stop", "全部跌破止损")]:
            d = t[k] - t["pnl"]
            L.append(f"  {lbl:<9}: 盈亏 {t[k]:>+11,.0f}円   (相对现在 {d:+,.0f}円)")
        L += ["", "  ↑「全部到T1」= 假设每只都涨到各自第一道压力位时的组合盈亏。",
              "    这是极端情景不是预测——实际不可能所有股票同时到位。",
              "", "── 逐只明细(点上方行看单只详情) ──"]
        for a in valid:
            L.append(f"  {a['ticker']:<9}{a['name'][:14]:<16}{a['pnl']:>+10,.0f}円"
                     f" ({a['pnl_pct']:+6.1f}%)   上行/下行 "
                     + (f"{a['ratio']:.2f}" if a.get("ratio") == a.get("ratio") else "-"))
        skipped = [rows[i]["ticker"] for i, a in enumerate(self._pf_results) if a is None]
        if skipped:
            L.append(f"\n  ⚠无数据(已跳过): {', '.join(skipped)}")
        self.txt_pf.setPlainText("\n".join(L))
        self.statusBar().showMessage("持仓测算完成")

    def _on_pos_select(self):
        """点某一行 → 显示该股的完整情景测算"""
        if not getattr(self, "_pf_results", None):
            return
        rows = {i.row() for i in self.tbl_pos.selectedItems()}
        if len(rows) != 1:
            return
        i = rows.pop()
        if i >= len(self._pf_results):
            return
        a = self._pf_results[i]
        if a is None:      # 该行标的无数据(池外且抓取失败)
            self.txt_pf.setPlainText(
                f"该标的无行情数据，无法测算。\n"
                f"可能原因：不在当前股票池(如ETF/其他市场)，或代码有误。")
            return
        lv = a.get("_levels")
        L = [f"══ {a['ticker']}  {a['name']}  [{a['kind']}] ══",
             f"  持有 {a['qty']:,}股   成本 {a['cost']:,.2f}円   现价 {a['price']:,.0f}円",
             f"  投入 {a['cost_amt']:,.0f}円 → 市值 {a['value']:,.0f}円   "
             f"当前盈亏 {a['pnl']:+,.0f}円 ({a['pnl_pct']:+.1f}%)", ""]
        if not a["scenarios"]:
            L.append("  (无法计算价位，可能数据不足)")
        else:
            L.append(f"  {'情景':<12}{'目标价':>10}{'需涨跌':>9}{'届时总盈亏':>14}{'盈亏%':>9}{'较现在':>13}")
            L.append("  " + "─" * 68)
            for s in a["scenarios"]:
                tag = "↑" if s["kind"] == "上行" else "↓"
                L.append(f"  {tag}{s['name']:<11}{s['price']:>10,.0f}{s['move_pct']:>+8.1f}%"
                         f"{s['pnl']:>+14,.0f}{s['pnl_pct']:>+8.1f}%{s['delta']:>+13,.0f}")
            if a.get("ratio") == a.get("ratio"):
                L += ["", f"  最大上行空间 {a['upside']:+,.0f}円   "
                          f"最大下行风险 {a['downside']:+,.0f}円   "
                          f"上行/下行 = {a['ratio']:.2f}"]
                L.append("  " + ("→ 上行空间大于下行风险" if a["ratio"] > 1
                                 else "→ ⚠下行风险大于上行空间，这个位置持有性价比不佳"))
        if lv:
            L += ["", f"  参考价位: 支撑 " + " / ".join(f"{x['price']:,.0f}" for x in lv["support"][:3])
                  + "    压力 " + " / ".join(f"{x['price']:,.0f}" for x in lv["resistance"][:3]),
                  f"  ATR(14) {lv['atr']:,.0f}円  日均波动 {lv['atr'] / lv['price']:.1%}"]
        L += ["", "⚠「届时总盈亏」是价格到达该位时的账面盈亏(含已有浮盈亏)，",
              "  「较现在」是相对当前市值的变化。目标位来自压力位，不是预测，",
              "  价格未必会到，也可能穿过。信用仓还需另计利息成本。"]
        self.txt_pf.setPlainText("\n".join(L))

    # ---------- 个股趋势诊断页 ----------
    def _tab_diagnosis(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        self.cb_dx = QtWidgets.QComboBox(); self.cb_dx.setEditable(True); self.cb_dx.setMinimumWidth(300)
        self.cb_dx.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.sp_dx_h = QtWidgets.QSpinBox(); self.sp_dx_h.setRange(1, 120); self.sp_dx_h.setValue(20)
        self.sp_dx_h.setSuffix(" 日")
        btn = QtWidgets.QPushButton("诊断这只股票")
        btn.setStyleSheet("font-weight:bold; padding:6px 18px;")
        btn.clicked.connect(self.on_diagnose)
        self.btn_fav = QtWidgets.QPushButton("☆ 收藏")
        self.btn_fav.setToolTip("加入/移出收藏，收藏保存在程序目录 watchlist.json")
        self.btn_fav.clicked.connect(self.on_toggle_fav)
        self.cb_fav = QtWidgets.QComboBox(); self.cb_fav.setMinimumWidth(260)
        self.cb_fav.setToolTip("收藏的股票，选中即诊断")
        self.cb_fav.activated.connect(self.on_pick_fav)
        bar.addWidget(QtWidgets.QLabel("股票:")); bar.addWidget(self.cb_dx)
        bar.addWidget(QtWidgets.QLabel("看未来:")); bar.addWidget(self.sp_dx_h)
        bar.addWidget(btn); bar.addWidget(self.btn_fav)
        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("★收藏:")); bar.addWidget(self.cb_fav)
        bar.addStretch()
        lay.addLayout(bar)

        self.lbl_verdict = QtWidgets.QLabel("请先加载数据，再选一只股票诊断")
        self.lbl_verdict.setStyleSheet(
            "font-size:19px; font-weight:bold; padding:12px; background:#f1f3f5; border-radius:6px;")
        lay.addWidget(self.lbl_verdict)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QTabWidget()
        # -- 判断清单
        p1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(p1)
        warn = QtWidgets.QLabel(
            "⚠ 这10项判据我用全市场98万个历史样本检验过：得分高的股票，之后20天上涨概率 52.9%，"
            "反而不如随便哪天买入的 54.6%。\n"
            "这个差距不是巧合(纯属运气的概率约万分之一)——日股是「涨多了会回调」的市场，"
            "而这10项全是「最近涨得好」的不同说法，方向天然反了。\n"
            "→ 只能当「现在是什么状态」看，别当买入信号。")
        warn.setWordWrap(True)
        warn.setStyleSheet("background:#fff3bf; padding:8px; border-radius:4px; font-size:12px;")
        l1.addWidget(warn)
        self.tbl_dx = QtWidgets.QTableWidget()
        self.tbl_dx.setColumnCount(5)
        self.tbl_dx.setHorizontalHeaderLabels(["", "判据", "权重", "得分", "实际数值"])
        l1.addWidget(self.tbl_dx)
        self.txt_dx = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_dx.setStyleSheet("font-family: Consolas, monospace; font-size:12px;")
        self.txt_dx.setMaximumHeight(170)
        l1.addWidget(self.txt_dx)
        left.addTab(p1, "判断清单")
        # -- 关键价位
        p2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(p2)
        how = QtWidgets.QLabel(
            "📐 这些价位是怎么算出来的(全部可复核,无主观成分)：\n"
            "· 压力位/支撑位 ← 两个来源合并：\n"
            "   ① 摆动高低点：左右各5根K线内的局部最高/最低价(近250日)，"
            "相近的(差<2%)合并成一条线\n"
            "   ② 成交量密集区：近250日成交量按价格分40档，量最大的档位(POC)——"
            "那里套牢盘和获利盘最多，天然形成阻力\n"
            "   现价之下的叫支撑，之上的叫压力\n"
            "· 「历史守住 x/y」← 历史上价格y次接近该线(±1.5%内)，"
            "其中x次在随后5日内没被有效突破(超出2%)。y太小或x/y太低=这条线不可靠\n"
            "· ATR(14) ← 近14日「真实波幅」均值，代表这只股票每天正常波动多少円\n"
            "· 买入参考 = 最近支撑位 × 1.005（回踩到支撑上方一点）\n"
            "· 止损位　 = 最近支撑位 − 2×ATR（放在日常波动之外，避免被随机噪音扫掉）\n"
            "· 目标位　 = 最近压力位\n"
            "· 盈亏比　 = (目标−买入) ÷ (买入−止损)，<1 表示冒的风险大于潜在收益\n"
            "⚠这套方法是技术分析的通行做法，但**未经本工具的统计验证**——"
            "支撑压力是市场参与者的行为惯性，不是物理定律。")
        how.setWordWrap(True)
        how.setStyleSheet("background:#f8f9fa; border:1px solid #dee2e6; padding:8px; "
                          "border-radius:4px; font-size:12px;")
        l2.addWidget(how)
        self.txt_lv = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_lv.setStyleSheet("font-family: Consolas, monospace; font-size:12px;")
        l2.addWidget(self.txt_lv)
        left.addTab(p2, "支撑/压力/买卖位")
        # -- 行业对比
        p3 = QtWidgets.QWidget(); l3 = QtWidgets.QVBoxLayout(p3)
        self.txt_sec = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_sec.setStyleSheet("font-family: Consolas, monospace; font-size:12px;")
        l3.addWidget(self.txt_sec)
        left.addTab(p3, "行业内对比")
        split.addWidget(left)

        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right)
        self.canvas_dx = Canvas(2, 1, (8, 6.4), interactive=True)
        rl.addWidget(NavToolbar(self.canvas_dx, self))
        tip = QtWidgets.QLabel(
            "🖱 滚轮=缩放(以光标为中心) · Shift+滚轮=只缩时间轴 · 左键拖动=平移 · "
            "⌨ R=重置 · ←→=平移 · ↑↓=缩放  (先点一下图再按键)")
        tip.setStyleSheet("color:#495057; font-size:11px; padding:2px;")
        rl.addWidget(tip)
        rl.addWidget(self.canvas_dx)
        split.addWidget(right)
        split.setSizes([560, 720])
        lay.addWidget(split)
        return w

    def on_diagnose(self):
        if self.panel is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在「数据」页加载面板")
            return
        tk = self._to_ticker(self.cb_dx.currentText())
        if tk not in self.prices.columns:
            QtWidgets.QMessageBox.warning(self, "提示", f"股票池中没有 {tk}")
            return
        self._sync_fav_button(tk)
        from diagnosis import diagnose
        vol = to_wide(self.panel, "volume")
        d = diagnose(self.prices[tk], vol[tk], horizon=self.sp_dx_h.value())
        if "error" in d:
            QtWidgets.QMessageBox.warning(self, "提示", d["error"])
            return

        color = {"上升趋势确立": "#2f9e44", "偏强/趋势形成中": "#66a80f", "横盘整理": "#f08c00",
                 "偏弱": "#e8590c", "下降趋势": "#c92a2a"}.get(d["verdict"], "#495057")
        self.lbl_verdict.setStyleSheet(
            f"font-size:19px; font-weight:bold; padding:12px; color:white; "
            f"background:{color}; border-radius:6px;")
        nm = getattr(self, "names", {}).get(tk, "")
        self.lbl_verdict.setText(
            f"{tk} {nm}   【{d['verdict']}】   {d['score']:.0f}/{d['total']} 分   "
            f"现价 {d['price']:,.0f}円   ({d['note']})")

        wts = d.get("weights", {})
        # 按权重从大到小排,让重要判据在最上面
        items = sorted(d["checks"].items(), key=lambda kv: -wts.get(kv[0], 0))
        self.tbl_dx.setRowCount(len(items))
        for i, (k, ok) in enumerate(items):
            wt = wts.get(k, 0)
            it = QtWidgets.QTableWidgetItem("✓" if ok else "✗")
            it.setForeground(QtCore.Qt.darkGreen if ok else QtCore.Qt.red)
            self.tbl_dx.setItem(i, 0, it)
            self.tbl_dx.setItem(i, 1, QtWidgets.QTableWidgetItem(k))
            w_it = QtWidgets.QTableWidgetItem(f"{wt}")
            w_it.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tbl_dx.setItem(i, 2, w_it)
            s_it = QtWidgets.QTableWidgetItem(f"{wt}" if ok else "0")
            s_it.setTextAlignment(QtCore.Qt.AlignCenter)
            if ok:
                f2 = s_it.font(); f2.setBold(True); s_it.setFont(f2)
                s_it.setForeground(QtCore.Qt.darkGreen)
            self.tbl_dx.setItem(i, 3, s_it)
            self.tbl_dx.setItem(i, 4, QtWidgets.QTableWidgetItem(d["detail"].get(k, "")))
        self.tbl_dx.resizeColumnsToContents()

        h, hs = d["horizon"], d["hist"]
        edge = (hs["胜率"] - hs["基准胜率"]) if hs["胜率"] == hs["胜率"] else float("nan")
        lines = [
            f"历史检验 — 这只股票过去出现「{d['strong_score']}项以上达标」时:",
            f"  样本数      : {hs['样本数']} 次",
            f"  {h}日后上涨概率: {hs['胜率']:.0%}" if hs["胜率"] == hs["胜率"] else "  样本不足",
            f"  {h}日平均收益  : {hs['平均收益']:+.2%}" if hs["平均收益"] == hs["平均收益"] else "",
            f"  中位数收益   : {hs['中位数收益']:+.2%}" if hs["中位数收益"] == hs["中位数收益"] else "",
            "",
            f"对照(该股任意时点买入): 胜率{hs['基准胜率']:.0%} 平均{hs['基准平均']:+.2%}",
            f"→ 强势形态相对基准的优势: 胜率{edge:+.0%}" if edge == edge else "",
            "",
            "⚠这是该股自身历史统计，样本重叠(相邻交易日高度相关)，",
            "  优势不明显时不要当作可靠信号。",
        ]
        self.txt_dx.setPlainText("\n".join(x for x in lines if x != ""))

        # 关键价位 + 行业对比
        lv = self._fill_levels(tk)
        self._fill_sector(tk)

        axes = self.canvas_dx.clear(2, 1)
        ax = axes[0]
        px, (ma5, ma20, ma60) = d["series"], d["ma"]
        show = px.index[-500:]
        ax.plot(show, px.reindex(show), lw=1.2, color="#212529", label="复权价")
        ax.plot(show, ma20.reindex(show), lw=1.0, color="#1971c2", label="MA20")
        ax.plot(show, ma60.reindex(show), lw=1.0, color="#e8590c", label="MA60")
        if lv:
            for x in lv["resistance"][:3]:
                ax.axhline(x["price"], color="#c92a2a", ls="--", lw=0.9, alpha=0.75)
                ax.annotate(f"压力 {x['price']:,.0f}", (show[-1], x["price"]), fontsize=7,
                            color="#c92a2a", xytext=(4, 0), textcoords="offset points", va="center")
            for x in lv["support"][:3]:
                ax.axhline(x["price"], color="#2f9e44", ls="--", lw=0.9, alpha=0.75)
                ax.annotate(f"支撑 {x['price']:,.0f}", (show[-1], x["price"]), fontsize=7,
                            color="#2f9e44", xytext=(4, 0), textcoords="offset points", va="center")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title(f"{tk} 近两年走势 · 均线 · 支撑压力位")
        ax = axes[1]
        sc = d["scores"].reindex(show)
        ax.fill_between(show, 0, sc, color="#4dabf7", alpha=0.6, step="mid")
        ax.axhline(d["strong_score"], color="#2f9e44", ls="--", lw=1.2, label=f"强势线({d['strong_score']}项)")
        ax.axhline(4, color="#f08c00", ls="--", lw=1.0, label="整理线(4项)")
        ax.set_ylim(0, 10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("趋势强度得分(0-10项) — 绿线以上=技术面强势状态")
        self.canvas_dx.draw()
        self.canvas_dx.save_home()      # 记住初始视图,供R键重置
        self.canvas_dx.setFocus()
        self.statusBar().showMessage(f"{tk} 诊断完成: {d['verdict']}")

    # ---------- 股票名 / 收藏 ----------
    def _label(self, ticker: str) -> str:
        """下拉显示: 7011.T  三菱重工業"""
        nm = getattr(self, "names", {}).get(ticker, "")
        return f"{ticker}  {nm}" if nm else ticker

    @staticmethod
    def _to_ticker(text: str) -> str:
        """从 '7011.T  三菱重工業' 或 '7011' 反解出 ticker"""
        t = (text or "").strip().split()[0] if text.strip() else ""
        if t and not t.endswith(".T") and t.replace("A", "").isdigit():
            t += ".T"
        return t

    def _refresh_fav(self):
        import watchlist
        favs = watchlist.load()
        self.cb_fav.blockSignals(True)
        self.cb_fav.clear()
        self.cb_fav.addItem(f"— 共{len(favs)}只 —")
        for t in favs:
            self.cb_fav.addItem(self._label(t), t)
        self.cb_fav.blockSignals(False)

    def on_toggle_fav(self):
        import watchlist
        tk = self._to_ticker(self.cb_dx.currentText())
        if not tk:
            return
        if watchlist.contains(tk):
            watchlist.remove(tk)
            self.statusBar().showMessage(f"已从收藏移除 {tk}")
        else:
            watchlist.add(tk)
            self.statusBar().showMessage(f"已收藏 {self._label(tk)}")
        self._refresh_fav()
        self._sync_fav_button(tk)

    def _sync_fav_button(self, tk: str):
        import watchlist
        on = watchlist.contains(tk)
        self.btn_fav.setText("★ 已收藏" if on else "☆ 收藏")
        self.btn_fav.setStyleSheet("color:#f08c00; font-weight:bold;" if on else "")

    def on_pick_fav(self, idx: int):
        tk = self.cb_fav.itemData(idx)
        if not tk:
            return
        self.cb_dx.setCurrentText(self._label(tk))
        self.on_diagnose()

    def _fill_levels(self, tk):
        """计算并显示支撑/压力/参考买卖位"""
        try:
            from levels import trade_levels
            hi = to_wide(self.panel, "high")[tk].dropna()
            lo = to_wide(self.panel, "low")[tk].dropna()
            cl = self.prices[tk].dropna()
            vo = to_wide(self.panel, "volume", scrub=False)[tk].dropna()
            r = trade_levels(hi, lo, cl, vo)
        except Exception as e:
            self.txt_lv.setPlainText(f"价位计算失败: {e}")
            return None
        L = [f"现价 {r['price']:,.0f}円   ATR(14) {r['atr']:,.0f}円 (日均波动 {r['atr'] / r['price']:.1%})",
             f"筹码密集价(POC) {r['poc']:,.0f}円" if r["poc"] == r["poc"] else "", "",
             "── 压力位(上方) ──"]
        for x in r["resistance"]:
            hold = f"{x['held']}/{x['tested']}" if x["tested"] else "-"
            L.append(f"  {x['price']:>9,.0f}  ({x['dist_pct']:+5.1f}%)  历史守住 {hold}"
                     + ("  ★筹码密集" if x["is_poc"] else ""))
        L += ["", "── 支撑位(下方) ──"]
        for x in r["support"]:
            hold = f"{x['held']}/{x['tested']}" if x["tested"] else "-"
            L.append(f"  {x['price']:>9,.0f}  ({x['dist_pct']:+5.1f}%)  历史守住 {hold}"
                     + ("  ★筹码密集" if x["is_poc"] else ""))
        L += ["", "── 机械参考位(算法输出,非建议) ──",
              f"  买入参考: {r['entry']:>9,.0f}   (最近支撑上方0.5%)",
              f"  止损位  : {r['stop']:>9,.0f}   ({r['stop_pct']:+.1f}%, 支撑下方2倍ATR)",
              f"  单笔风险: {r['risk']:>9,.0f}円/股  (买入价−止损价)", "",
              f"  目标(来源:{r['target_src']}):"]
        for t in r.get("targets", []):
            flag = "  ✓盈亏比达标" if t["rr"] >= 1 else ""
            flag = "  ★盈亏比优秀" if t["rr"] >= 2 else flag
            L.append(f"    {t['name']}: {t['price']:>9,.0f}  ({t['pct']:+5.1f}%)"
                     f"  盈亏比 {t['rr']:.2f}{flag}")
        ok, good = r.get("rr_ok_target"), r.get("rr_good_target")
        if good:
            L.append(f"  → 涨到 {good['name']}({good['price']:,.0f}) 时盈亏比才达2.0，"
                     f"这是「值得冒险」的位置")
        elif ok:
            L.append(f"  → 涨到 {ok['name']}({ok['price']:,.0f}) 时盈亏比才够1.0，"
                     f"T1之前都是「赚得比亏得少」")
        else:
            L.append("  → ⚠所有目标的盈亏比都<1：按这个止损位，潜在收益盖不住风险")
        L += ["",
              "【怎么读这几个数】",
              "· T1(第一道压力)离得近是必然的——它是「价格可能卡住的地方」，不是盈利目标。",
              "  真正该问的是:「盈亏比≥1 要涨到第几档」。若T3才够，说明这笔要拿很久。",
              "· 盈亏比 = 潜在赚幅 ÷ 潜在亏幅。<1 = 赌赢了赚得还没赌输了亏得多。",
              "· 止损放在支撑下方2倍ATR(日常波动之外)，所以止损幅度看着大——",
              "  这是为了不被随机波动扫出局，代价就是盈亏比被压低。",
              "  想要更高盈亏比，要么等价格更靠近支撑再买，要么接受更紧(更易被扫)的止损。",
              "· 「历史守住 x/y」= 价格曾y次接近该位，其中x次未被有效突破；y太小=样本不足不可信。",
              "⚠支撑压力是经验规律不是定律，以上均为算法输出，不构成建议。"]
        self.txt_lv.setPlainText("\n".join(x for x in L if x != ""))
        return r

    def _fill_sector(self, tk):
        """行业内相对表现"""
        try:
            from sector_rel import peer_snapshot
            if self.sector is None:
                self.txt_sec.setPlainText("行业数据不可用")
                return
            ps = peer_snapshot(self.prices, self.sector, tk)
        except Exception as e:
            self.txt_sec.setPlainText(f"行业对比失败: {e}")
            return
        if "error" in ps:
            self.txt_sec.setPlainText(ps["error"])
            return
        n = ps["n_peers"]
        L = [f"所属行业: {ps['sector']}   同业 {n} 只", ""]
        L.append(f"{'周期':<10}{'本股':>10}{'行业中位':>10}{'超额':>10}{'行业排名':>12}")
        L.append("-" * 54)
        for c in ps["table"].columns:
            s = ps["self"][c] if ps["self"] is not None else float("nan")
            m = ps["sector_median"][c]
            rk = ps["ranks"].get(c)
            L.append(f"{c:<10}{s:>+10.1%}{m:>+10.1%}{s - m:>+10.1%}"
                     + (f"{f'第{rk}/{n}':>12}" if rk else f"{'-':>12}"))
        L += ["", "── 行业内近60日领先者 ──"]
        for t, row in ps["leaders"].iterrows():
            mark = " ←本股" if t == tk else ""
            nm = getattr(self, "names", {}).get(t, "")
            L.append(f"  {t:<9}{row.get('60日涨幅', float('nan')):+8.1%}  {nm}{mark}")
        L += ["", "超额>0 = 跑赢同行(个股自身因素占优); 超额<0 = 跑输同行。",
              "这剥离了板块β,比看绝对涨跌更能说明这只股票本身强弱。"]
        self.txt_sec.setPlainText("\n".join(L))

    # ---------- 回测页 ----------
    def _tab_backtest(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGridLayout()
        self.cb_factor = QtWidgets.QComboBox()
        self.cb_factor.addItems(FACTORS.keys())
        self.sp_window = QtWidgets.QSpinBox(); self.sp_window.setRange(2, 250); self.sp_window.setValue(20)
        self.sp_lookback = QtWidgets.QSpinBox(); self.sp_lookback.setRange(20, 500); self.sp_lookback.setValue(245)
        self.sp_skip = QtWidgets.QSpinBox(); self.sp_skip.setRange(0, 60); self.sp_skip.setValue(20)
        self.sp_horizon = QtWidgets.QSpinBox(); self.sp_horizon.setRange(1, 120); self.sp_horizon.setValue(20)
        self.sp_groups = QtWidgets.QSpinBox(); self.sp_groups.setRange(3, 10); self.sp_groups.setValue(5)
        self.chk_sector = QtWidgets.QCheckBox("行业中性")
        self.chk_size = QtWidgets.QCheckBox("规模中性")
        self.sp_spread = QtWidgets.QDoubleSpinBox(); self.sp_spread.setRange(0, 100); self.sp_spread.setValue(5.0)
        self.sp_comm = QtWidgets.QDoubleSpinBox(); self.sp_comm.setRange(0, 100); self.sp_comm.setValue(0.0)
        items = [("因子:", self.cb_factor), ("窗口:", self.sp_window), ("回看:", self.sp_lookback),
                 ("跳过:", self.sp_skip), ("持有期:", self.sp_horizon), ("分组数:", self.sp_groups),
                 ("价差bp:", self.sp_spread), ("手续费bp:", self.sp_comm)]
        for i, (lbl, wid) in enumerate(items):
            g.addWidget(QtWidgets.QLabel(lbl), i // 4, (i % 4) * 2)
            g.addWidget(wid, i // 4, (i % 4) * 2 + 1)
        g.addWidget(self.chk_sector, 2, 0, 1, 2)
        g.addWidget(self.chk_size, 2, 2, 1, 2)
        self.btn_run = QtWidgets.QPushButton("运行回测")
        self.btn_run.clicked.connect(self.on_run)
        g.addWidget(self.btn_run, 2, 6, 1, 2)
        lay.addLayout(g)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.canvas_bt = Canvas(1, 3, (13, 4.2))
        top = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(top)
        tl.addWidget(NavToolbar(self.canvas_bt, self)); tl.addWidget(self.canvas_bt)
        split.addWidget(top)
        self.txt_bt = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_bt.setStyleSheet("font-family: Consolas, monospace;")
        split.addWidget(self.txt_bt)
        split.setSizes([460, 300])
        lay.addWidget(split)
        return w

    def _make_factor(self):
        name = self.cb_factor.currentText()
        p = {"window": self.sp_window.value(), "lookback": self.sp_lookback.value(),
             "skip": self.sp_skip.value()}
        f = FACTORS[name](self.prices, p)
        tag = name
        if self.chk_sector.isChecked() or self.chk_size.isChecked():
            f = neutralize(f, sector=self.sector if self.chk_sector.isChecked() else None,
                           size=self.size if self.chk_size.isChecked() else None)
            tag += "-" + ("行业" if self.chk_sector.isChecked() else "") + \
                   ("规模" if self.chk_size.isChecked() else "") + "中性"
        return f, tag

    def on_run(self):
        if self.prices is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在「数据」页加载面板")
            return
        self.btn_run.setEnabled(False)
        self.statusBar().showMessage("回测中…")
        h, ng = self.sp_horizon.value(), self.sp_groups.value()
        sp, cm = self.sp_spread.value(), self.sp_comm.value()

        def job():
            f, tag = self._make_factor()
            fa, ra = align(f, forward_returns(self.prices, h), self.mask)
            ic = compute_ic(fa, ra)
            stats = ic_summary(ic, h)
            qr = quantile_returns(fa, self.prices.pct_change(), ng, h)
            decay = ic_decay(f, self.prices, mask=self.mask)
            crep = costs_mod.cost_report(qr, fa, ng, h, spread_bp=sp, commission_bp=cm)
            ls_ann = crep.get("多空毛年化", float("nan"))
            ls_shp = crep.get("毛夏普", float("nan"))
            research_log.log_trial(tag, stats, universe=self.cb_market.currentText(),
                                   horizon=h, rebalance=h,
                                   period=f"{fa.index.min().date()}~{fa.index.max().date()}",
                                   ls_ann=ls_ann, ls_sharpe=ls_shp, note="GUI")
            return dict(tag=tag, ic=ic, stats=stats, qr=qr, decay=decay, crep=crep,
                        ng=ng, h=h, factor=f)

        self.worker = Worker(job)
        self.worker.done.connect(self._ran)
        self.worker.failed.connect(self._err)
        self.worker.start()

    def _ran(self, res):
        self.last_result = res
        ic, qr, decay, ng = res["ic"], res["qr"], res["decay"], res["ng"]
        axes = self.canvas_bt.clear(1, 3)
        ax = axes[0]
        ic.plot(ax=ax, lw=0.6, alpha=0.5)
        ic.rolling(60, min_periods=20).mean().plot(ax=ax, lw=1.8, color="#e03131")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"IC时序 均值{res['stats']['IC均值']:.4f} t={res['stats']['t统计量']:.2f}")
        ax = axes[1]
        for c in qr.columns:
            if c.startswith("Q") and "-" not in c:
                (1 + qr[c]).cumprod().plot(ax=ax, lw=1.1, label=c)
        ax.set_yscale("log"); ax.legend(fontsize=7); ax.set_title("分组累计净值")
        ax = axes[2]
        decay.plot(ax=ax, marker="o"); ax.axhline(0, color="k", lw=0.8)
        ax.set_title("IC衰减"); ax.set_xlabel("持有期(日)")
        self.canvas_bt.draw()

        import plain
        n = len(research_log.load_trials())
        t = res["stats"]["t统计量"]
        ann = (1 + qr.mean()) ** TRADING_DAYS - 1
        L = [f"因子: {res['tag']}   持有期{res['h']}日 {ng}分组", "=" * 70,
             "【一句话结论】", f"  {plain.explain_t(t, '这个因子', '数值越高,之后越涨', '数值越高,之后反而越跌')}",
             f"  {plain.coin_flip_analogy(t)}",
             f"  {plain.verdict_line(t, n)}", "",
             "【详细数字】"]
        L += [f"  {k:<12}: {v:>10.4f}" if isinstance(v, float) else f"  {k:<12}: {v:>10}"
              for k, v in res["stats"].items()]
        L.append("    ↑ IC均值=预测准确度(0=没用,±0.03以上算不错) ICIR=信号稳定性 "
                 "t统计量=可信度(见上方人话解释)")
        L.append("\n  分组年化: " + " | ".join(f"{c}:{ann[c]:+.1%}" for c in qr.columns))
        L.append("    ↑ 按因子值把股票分5组,Q1最低Q5最高。若因子有用,应呈单调排列")
        L.append("\n  ── 交易成本检验(能不能真赚到) ──")
        for k, v in res["crep"].items():
            L.append(f"  {k:<14}: " + (f"{v:+.2%}" if ("年化" in k and "次数" not in k and "换手" not in k)
                                       else f"{v:.2f}"))
        be = res["crep"].get("盈亏平衡成本bp")
        if be == be:
            L.append(f"    ↑ 盈亏平衡成本{be:.1f}bp = 每次买卖成本超过 {be / 100:.3f}% 就白干。"
                     f"日股实际买卖价差通常0.05%~0.1%")
        self.txt_bt.setPlainText("\n".join(L))
        self.btn_run.setEnabled(True)
        self.statusBar().showMessage("回测完成")
        self._refresh_log()

    # ---------- 信号点位页 ----------
    def _tab_signal(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        self.cb_ticker = QtWidgets.QComboBox(); self.cb_ticker.setEditable(True)
        self.cb_ticker.setMinimumWidth(140)
        self.sp_top = QtWidgets.QDoubleSpinBox(); self.sp_top.setRange(1, 50); self.sp_top.setValue(20)
        self.sp_top.setSuffix(" %")
        self.sp_gap = QtWidgets.QSpinBox(); self.sp_gap.setRange(1, 60); self.sp_gap.setValue(5)
        self.btn_sig = QtWidgets.QPushButton("标注信号点位")
        self.btn_sig.clicked.connect(self.on_signal)
        for lbl, wid in [("股票:", self.cb_ticker), ("排名前:", self.sp_top), ("最小间隔(日):", self.sp_gap)]:
            bar.addWidget(QtWidgets.QLabel(lbl)); bar.addWidget(wid)
        bar.addWidget(self.btn_sig); bar.addStretch()
        lay.addLayout(bar)
        lay.addWidget(QtWidgets.QLabel(
            "用当前「因子回测」页的因子设置；▲=信号点，绿色表示其后持有期内上涨、红色下跌"))
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.canvas_sig = Canvas(1, 1, (13, 5))
        top = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(top)
        tl.addWidget(NavToolbar(self.canvas_sig, self)); tl.addWidget(self.canvas_sig)
        split.addWidget(top)
        self.tbl_sig = QtWidgets.QTableWidget()
        split.addWidget(self.tbl_sig)
        split.setSizes([520, 260])
        lay.addWidget(split)
        return w

    def on_signal(self):
        if self.prices is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载面板")
            return
        tk = self._to_ticker(self.cb_ticker.currentText())
        if tk not in self.prices.columns:
            QtWidgets.QMessageBox.warning(self, "提示", f"股票池中没有 {tk}")
            return
        f, tag = self._make_factor()
        h = self.sp_horizon.value()
        pts = signal_points(f, tk, top_pct=self.sp_top.value() / 100,
                            mask=self.mask, min_gap=self.sp_gap.value())
        pts = forward_outcome(self.prices, tk, pts, horizon=h)

        ax = self.canvas_sig.clear(1, 1)
        px = self.prices[tk].dropna()
        ax.plot(px.index, px.values, lw=1.0, color="#1971c2", label=f"{tk} 复权价")
        col = f"{h}日后收益"
        if not pts.empty:
            for _, r in pts.iterrows():
                y = px.get(r["date"], np.nan)
                if y != y:
                    continue
                ret = r.get(col, np.nan)
                c = "#2f9e44" if (ret == ret and ret > 0) else ("#e03131" if ret == ret else "#868e96")
                ax.scatter(r["date"], y, marker="^", s=70, color=c, zorder=5, edgecolors="white", lw=0.6)
        wins = (pts[col] > 0).sum() if (not pts.empty and col in pts) else 0
        tot = pts[col].notna().sum() if (not pts.empty and col in pts) else 0
        ax.set_title(f"{tk} — {tag} 信号点位 (共{len(pts)}个, 已知结果{tot}个, "
                     f"胜率{wins / tot:.0%})" if tot else f"{tk} — {tag} 信号点位 (共{len(pts)}个)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        self.canvas_sig.draw()

        self.tbl_sig.clear()
        cols = ["date", "rank_pct", "factor"] + ([col] if col in pts.columns else [])
        self.tbl_sig.setColumnCount(len(cols)); self.tbl_sig.setRowCount(len(pts))
        self.tbl_sig.setHorizontalHeaderLabels(["日期", "截面排名", "因子值", f"{h}日后收益"][:len(cols)])
        for i, (_, r) in enumerate(pts.iterrows()):
            vals = [str(pd.Timestamp(r["date"]).date()), f"{r['rank_pct']:.1%}", f"{r['factor']:.4f}"]
            if col in pts.columns:
                v = r[col]
                vals.append(f"{v:+.2%}" if v == v else "-")
            for j, v in enumerate(vals):
                self.tbl_sig.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self.tbl_sig.resizeColumnsToContents()
        self.statusBar().showMessage(f"{tk}: {len(pts)}个信号点")

    # ---------- 登记册页 ----------
    def _tab_log(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        bar = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("刷新")
        btn.clicked.connect(self._refresh_log)
        bar.addWidget(btn); bar.addStretch()
        lay.addLayout(bar)
        self.tbl_log = QtWidgets.QTableWidget()
        lay.addWidget(self.tbl_log)
        self.lbl_thr = QtWidgets.QLabel()
        self.lbl_thr.setStyleSheet("font-family: Consolas, monospace; padding:6px;")
        self.lbl_thr.setWordWrap(True)
        lay.addWidget(self.lbl_thr)
        self._refresh_log()
        return w

    def _refresh_log(self):
        trials = research_log.load_trials()
        if not trials:
            self.lbl_thr.setText("登记册为空")
            return
        import plain
        cols = ["因子名", "持有期", "IC均值", "t统计量", "这个数字什么意思", "ICIR年化", "多空夏普"]
        self.tbl_log.clear()
        self.tbl_log.setColumnCount(len(cols)); self.tbl_log.setRowCount(len(trials))
        self.tbl_log.setHorizontalHeaderLabels(cols)

        def _t(r):
            try:
                return abs(float(r["t统计量"]))
            except (ValueError, TypeError, KeyError):
                return 0.0
        for i, r in enumerate(sorted(trials, key=lambda x: -_t(x))):
            try:
                tv = float(r["t统计量"])
            except (ValueError, TypeError, KeyError):
                tv = 0.0
            meaning = (f"{plain.confidence_label(tv)}"
                       f"({'正向' if tv > 0 else '反向'},巧合概率{plain.odds_text(plain.p_value(tv))})"
                       if abs(tv) >= 1.5 else "看不出规律")
            vals = [r.get("因子名", ""), r.get("持有期", ""), r.get("IC均值", ""),
                    r.get("t统计量", ""), meaning, r.get("ICIR年化", ""), r.get("多空夏普", "")]
            for j, v in enumerate(vals):
                self.tbl_log.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))
        self.tbl_log.resizeColumnsToContents()
        n = len(trials)
        exp_max, bonf = research_log.expected_max_t(n), research_log.bonferroni_t(n)
        thr = max(bonf, 3.0)
        passed = [r["因子名"] for r in trials if _t(r) > thr]
        self.lbl_thr.setText(
            f"你一共试了 {n} 个因子(含调参数的变体)。\n"
            f"为什么试得越多、要求越严：纯靠运气，试{n}次里最好的那个"
            f"「t统计量」也能达到 {exp_max:.1f} 左右——所以合格线要提到 {thr:.1f} 以上，"
            f"才排除得掉「碰巧撞对」。\n"
            f"目前通过这条线的: {', '.join(sorted(set(passed))) if passed else '无'}\n"
            f"⚠ 调参数、换股票池、换持有期都算一次试验。记漏了 = 高估自己的发现。")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
