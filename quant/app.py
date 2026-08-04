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
from PyQt5 import QtCore, QtWidgets

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
    def __init__(self, nrows=1, ncols=1, figsize=(11, 4.5)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.axes = self.fig.subplots(nrows, ncols)

    def clear(self, nrows=1, ncols=1):
        self.fig.clear()
        self.axes = self.fig.subplots(nrows, ncols)
        return self.axes


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
        tabs.addTab(self._tab_backtest(), "④ 因子回测(研究用)")
        tabs.addTab(self._tab_signal(), "⑤ 信号点位")
        tabs.addTab(self._tab_log(), "⑥ 试验登记册")
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
        self.sp_minscore = QtWidgets.QSpinBox(); self.sp_minscore.setRange(1, 10); self.sp_minscore.setValue(8)
        self.sp_minscore.setSuffix(" 分以上")
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
        self.tree_scr.setColumnCount(7)
        self.tree_scr.setHeaderLabels(["领域 / 股票", "公司名", "得分", "现价",
                                       "20日涨幅", "同行超额", "60日涨幅"])
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
                       mode=self.cb_mode.currentText(), min_excess=self.sp_exc.value() / 100)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "扫描失败", traceback.format_exc()[-1200:])
            self.lbl_scan.setText("扫描失败")
            return

        self.tree_scr.clear()
        for sector, rows in res.items():
            top = QtWidgets.QTreeWidgetItem([f"【{sector}】", f"{len(rows)}只入选",
                                             "", "", "", "", ""])
            f = top.font(0); f.setBold(True); top.setFont(0, f)
            top.setBackground(0, QtCore.Qt.lightGray)
            for r in rows:
                nm = getattr(self, "names", {}).get(r["ticker"], "")
                child = QtWidgets.QTreeWidgetItem([
                    r["ticker"], nm, f"{r['score']}分", f"{r['price']:,.0f}",
                    f"{r['chg20']:+.1%}", f"{r['excess20']:+.1%}", f"{r['chg60']:+.1%}"])
                child.setData(0, QtCore.Qt.UserRole, r["ticker"])
                # 超额为正标红(相对同行强)，为负标绿
                child.setForeground(5, QtCore.Qt.red if r["excess20"] > 0 else QtCore.Qt.darkGreen)
                # 20日涨幅过大(>20%)标橙提示回调风险
                if r["chg20"] > 0.20:
                    child.setForeground(4, QtCore.Qt.darkYellow)
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
        self.tbl_dx.setColumnCount(3)
        self.tbl_dx.setHorizontalHeaderLabels(["", "判据", "实际数值"])
        l1.addWidget(self.tbl_dx)
        self.txt_dx = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt_dx.setStyleSheet("font-family: Consolas, monospace; font-size:12px;")
        self.txt_dx.setMaximumHeight(170)
        l1.addWidget(self.txt_dx)
        left.addTab(p1, "判断清单")
        # -- 关键价位
        p2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(p2)
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
        self.canvas_dx = Canvas(2, 1, (8, 6.4))
        rl.addWidget(NavToolbar(self.canvas_dx, self)); rl.addWidget(self.canvas_dx)
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
            f"{tk} {nm}   【{d['verdict']}】   {d['score']}/{d['total']} 项达标   "
            f"现价 {d['price']:,.0f}円   ({d['note']})")

        self.tbl_dx.setRowCount(len(d["checks"]))
        for i, (k, ok) in enumerate(d["checks"].items()):
            it = QtWidgets.QTableWidgetItem("✓" if ok else "✗")
            it.setForeground(QtCore.Qt.darkGreen if ok else QtCore.Qt.red)
            self.tbl_dx.setItem(i, 0, it)
            self.tbl_dx.setItem(i, 1, QtWidgets.QTableWidgetItem(k))
            self.tbl_dx.setItem(i, 2, QtWidgets.QTableWidgetItem(d["detail"].get(k, "")))
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
        ax.set_title("趋势强度得分(0-10项) — 绿线以上=上升趋势")
        self.canvas_dx.draw()
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
              f"  买入参考: {r['entry']:,.0f}   (最近支撑上方0.5%)",
              f"  止损位  : {r['stop']:,.0f}   ({r['stop_pct']:+.1f}%, 支撑下方2倍ATR)",
              f"  目标位  : {r['target']:,.0f}   ({r['target_pct']:+.1f}%, 最近压力)",
              f"  盈亏比  : {r['risk_reward']:.2f}" if r["risk_reward"] == r["risk_reward"] else "",
              "",
              "「历史守住 x/y」= 价格曾y次接近该位,其中x次未被有效突破。",
              "y次数少或x/y比例低的线不可靠。盈亏比<1表示风险大于潜在收益。",
              "⚠支撑压力是经验规律不是定律,机械参考位仅为算法输出,不构成建议。"]
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
