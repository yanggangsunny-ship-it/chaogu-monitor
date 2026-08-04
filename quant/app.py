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
        tabs.addTab(self._tab_data(), "① 数据")
        tabs.addTab(self._tab_backtest(), "② 因子回测")
        tabs.addTab(self._tab_signal(), "③ 信号点位")
        tabs.addTab(self._tab_log(), "④ 试验登记册")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("就绪 — 请先在「数据」页加载面板")

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

    def on_load(self):
        self.btn_load.setEnabled(False)
        self.statusBar().showMessage("下载/加载中…")
        mk, yr, rf = self.cb_market.currentText(), self.sp_years.value(), self.chk_refresh.isChecked()
        self.worker = Worker(build_panel, market=mk, years=yr, refresh=rf)
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
        self.cb_ticker.clear()
        self.cb_ticker.addItems(list(self.prices.columns))
        self.txt_data.appendPlainText(
            f"\n[已加载] {self.prices.shape[1]}只 × {self.prices.shape[0]}交易日 "
            f"| {self.prices.index.min().date()} ~ {self.prices.index.max().date()}"
            f"\n流动性过滤后日均可交易 {self.mask.sum(axis=1).mean():.0f}只"
            f" | 中性化数据: {'就绪' if self.sector is not None else '不可用'}")
        self.btn_load.setEnabled(True)
        self.statusBar().showMessage("面板加载完成")

    def _err(self, tb):
        self.btn_load.setEnabled(True)
        self.btn_run.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "出错", tb[-1500:])
        self.statusBar().showMessage("出错")

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

        n = len(research_log.load_trials())
        thr = max(research_log.bonferroni_t(n), 3.0)
        verdict = "✓ 站得住" if abs(res["stats"]["t统计量"]) > thr else "✗ 校正后不显著"
        ann = (1 + qr.mean()) ** TRADING_DAYS - 1
        L = [f"因子: {res['tag']}   持有期{res['h']}日 {ng}分组", "=" * 68]
        L += [f"  {k:<12}: {v:>10.4f}" if isinstance(v, float) else f"  {k:<12}: {v:>10}"
              for k, v in res["stats"].items()]
        L.append("\n  分组年化: " + " | ".join(f"{c}:{ann[c]:+.1%}" for c in qr.columns))
        L.append("\n  ── 成本分析 ──")
        for k, v in res["crep"].items():
            L.append(f"  {k:<14}: " + (f"{v:+.2%}" if ("年化" in k and "次数" not in k and "换手" not in k)
                                       else f"{v:.2f}"))
        L.append(f"\n  [登记册] 第{n}次试验 | 噪声地板t={research_log.expected_max_t(n):.2f}"
                 f" | 门槛t>{thr:.2f} | {verdict}")
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
        tk = self.cb_ticker.currentText().strip()
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
        cols = ["因子名", "持有期", "IC均值", "t统计量", "ICIR年化", "多空夏普", "试验时间"]
        self.tbl_log.clear()
        self.tbl_log.setColumnCount(len(cols)); self.tbl_log.setRowCount(len(trials))
        self.tbl_log.setHorizontalHeaderLabels(cols)

        def _t(r):
            try:
                return abs(float(r["t统计量"]))
            except (ValueError, TypeError, KeyError):
                return 0.0
        for i, r in enumerate(sorted(trials, key=lambda x: -_t(x))):
            for j, c in enumerate(cols):
                self.tbl_log.setItem(i, j, QtWidgets.QTableWidgetItem(str(r.get(c, ""))))
        self.tbl_log.resizeColumnsToContents()
        n = len(trials)
        exp_max, bonf = research_log.expected_max_t(n), research_log.bonferroni_t(n)
        thr = max(bonf, 3.0)
        passed = [r["因子名"] for r in trials if _t(r) > thr]
        self.lbl_thr.setText(
            f"累计 {n} 次试验   |   噪声地板(期望最大t) {exp_max:.2f}  ·  "
            f"Bonferroni {bonf:.2f}  ·  HLZ经验值 3.00   →   判定门槛 t > {thr:.2f}\n"
            f"通过的因子: {', '.join(sorted(set(passed))) if passed else '无'}\n"
            f"⚠调参数/换股票池/换持有期都算试验。少记 = 高估显著性。")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
