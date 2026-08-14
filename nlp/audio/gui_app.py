#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频采集与分析桌面应用 (PySide6, LGPL 许可，可商用，无版权问题)

功能:
  1. 设备扫描: 列出本机所有输入/输出音频设备并在页面上选择
  2. 录音: 可选不同时长 (5s/10s/30s/1min/3min/自定义) 录制
  3. 保存: 支持 wav / flac / ogg / mp3 多种格式保存
  4. 分析: 对指定目录的音频文件进行时域/频域/语谱图/Mel/MFCC/基频/降噪分析
  5. 对比: 多文件波形/频谱/MFCC/相似度矩阵对比

运行:
  pip install -r requirements.txt
  python gui_app.py

界面设计:
  窗口布局由 main_window.ui (Qt Designer 设计) 描述，可用 Qt Designer
  直接打开修改控件/排版，无需改动本文件；本文件仅负责业务逻辑与信号连接。
"""

import os
import sys
import time
import wave
import threading
import datetime as dt
from functools import partial

import numpy as np

# ---- 可选依赖优雅降级 ----
try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    sd = None
    _sd_err = e

try:
    import soundfile as sf
except Exception:
    sf = None

# Qt 子模块分开导入，避免任意一个失败导致后续顶层类 (如 class Recorder(QThread))
# 因名字未定义而抛出 NameError。失败时定义占位对象，让模块仍可 import，
# 真正的错误会在 main() 中清晰提示。
HAS_QT = True
_qt_err = None


class _Missing:
    """依赖缺失时的占位基类：让顶层 `class X(Y)` 仍可解析（不会 NameError），
       真实运行时实例化会抛出清晰错误，由 main() 统一拦截提示。"""
    def __init__(self, *a, **k):
        raise RuntimeError("依赖未正确安装，无法实例化组件")


def _require_qt(name):
    """导入失败时报一次明确错误，并标记 HAS_QT=False。"""
    global HAS_QT, _qt_err
    HAS_QT = False
    _qt_err = _qt_err or name


try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QComboBox, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
        QFileDialog, QListWidget, QCheckBox, QTabWidget, QTextEdit, QMessageBox,
        QProgressBar, QGroupBox, QSplitter, QAbstractItemView,
        QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QDialogButtonBox,
    )
    from PySide6.QtUiTools import QUiLoader  # 用于运行时加载 .ui 文件
except Exception as e:
    _require_qt("PySide6.QtWidgets"); _qt_err = e

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, QFile, QIODevice
except Exception as e:
    _require_qt("PySide6.QtCore"); _qt_err = _qt_err or e

try:
    from PySide6.QtGui import QIcon, QFont
except Exception as e:
    _require_qt("PySide6.QtGui"); _qt_err = _qt_err or e

# 占位：当 Qt 未正确安装时，让顶层类定义仍可解析（实际运行会在 main() 被拦截）
if not HAS_QT:
    QApplication = QMainWindow = QWidget = QVBoxLayout = QHBoxLayout = _Missing
    QPushButton = QComboBox = QLabel = QLineEdit = QSpinBox = _Missing
    QDoubleSpinBox = QFileDialog = QListWidget = QCheckBox = _Missing
    QTabWidget = QTextEdit = QMessageBox = QProgressBar = _Missing
    QGroupBox = QSplitter = QAbstractItemView = _Missing
    Qt = QTimer = QThread = QObject = _Missing
    # Signal 必须是“可调用且返回无害值”的占位：类体内 `sig = Signal(float)` 会在类定义时立即调用，
    # 不能让它 raise，否则整段 import 失败。真正的错误在 main() 拦截。
    Signal = lambda *a, **k: None
    QIcon = QFont = _Missing

    def _load_ui(ui_path, baseinstance=None):
        raise RuntimeError("PySide6 未安装，无法加载 UI")
else:
    def _load_ui(ui_path, baseinstance=None):
        """加载 .ui 文件 (Qt Designer 设计) 并挂载到 baseinstance (MainWindow)。

        用 QUiLoader 解析出 .ui 中的顶层 QMainWindow (form)，
        再将其 centralwidget 通过 takeCentralWidget 转移到 baseinstance 上。
        这样既保留 Designer 的布局，又正确设置了 central widget（避免窗口空白）。
        """
        from PySide6 import QtCore
        loader = QUiLoader()
        ui_file = QtCore.QFile(ui_path)
        ui_file.open(QtCore.QIODevice.ReadOnly)
        form = loader.load(ui_file, None)
        ui_file.close()
        if form is None:
            raise RuntimeError(f"加载 .ui 失败: {ui_path}")
        # 把 .ui 里 QMainWindow 的中央控件转移到 baseinstance
        cw = form.takeCentralWidget()
        if cw is not None:
            baseinstance.setCentralWidget(cw)
        # 同步标题/几何等顶层属性
        baseinstance.setWindowTitle(form.windowTitle())
        form.deleteLater()

try:
    import matplotlib
    matplotlib.use("QtAgg")  # 嵌入 PySide6 窗口
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    matplotlib = None
    plt = None
    FigureCanvas = object     # 占位基类，保证 class MplCanvas(FigureCanvas) 可解析且不炸
    Figure = object

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False

try:
    import librosa, librosa.display
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac")


# =========================================================================
# 音频核心逻辑 (与 GUI 解耦，便于测试/复用)
# =========================================================================
def list_audio_devices():
    """返回 (all_devices, input_devices, default_input_idx)"""
    if sd is None:
        return [], [], None
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    all_d, ins = [], []
    for i, d in enumerate(devs):
        api = hostapis[d["hostapi"]]["name"] if d["hostapi"] >= 0 else "-"
        rec = {
            "index": i,
            "name": d["name"],
            "hostapi": api,
            "max_in": d["max_input_channels"],
            "max_out": d["max_output_channels"],
            "default_sr": d["default_samplerate"],
            "is_input": d["max_input_channels"] > 0,
        }
        all_d.append(rec)
        if rec["is_input"]:
            ins.append(rec)
    default_idx = sd.default.device[0]
    return all_d, ins, default_idx


def list_output_devices():
    """返回 (output_devices, default_output_idx)。供试听选择输出设备。"""
    if sd is None:
        return [], None
    devs = sd.query_devices()
    outs = []
    for i, d in enumerate(devs):
        if d["max_output_channels"] > 0:
            outs.append({
                "index": i,
                "name": d["name"],
                "max_out": d["max_output_channels"],
                "default_sr": d["default_samplerate"],
            })
    default_out = sd.default.device[1] if sd.default.device[1] >= 0 else None
    return outs, default_out


def load_audio(path, target_sr=None):
    data, sr = sf.read(path, always_2d=False, dtype="float32")
    # 强制转成一维 mono，避免后续对多维数组做布尔判断报错
    data = np.asarray(data, dtype=float)
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data
    mono = np.asarray(mono, dtype=float).ravel()  # 确保一维
    if target_sr is not None and sr != target_sr and HAS_LIBROSA:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return data, mono, sr


def encode_mp3(data, sr):
    try:
        import lameenc
    except Exception as e:
        raise RuntimeError("未安装 lameenc（pip install lameenc）无法进行 MP3 编码") from e
    enc = lameenc.Encoder()
    enc.set_bit_rate(192)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1 if data.ndim == 1 else 2)
    enc.set_quality(2)
    pcm = data if data.dtype == np.int16 else (np.clip(data, -1, 1) * 32767).astype(np.int16)
    return enc.encode(pcm.tobytes())


def save_audio(data, sr, path, fmt):
    if fmt == "mp3":
        with open(path, "wb") as f:
            f.write(encode_mp3(data, sr))
    else:
        sf.write(path, data, sr, format=fmt.upper())


# =========================================================================
# 音频指标计算 (查看器页复用，纯函数便于测试)
# =========================================================================
def compute_audio_metrics(data, sr):
    """计算一组常用音频指标，返回 dict。

    data: 原始音频数组 (float)，sr: 采样率。
    返回指标: 采样率/声道/时长/样本数/RMS/峰值/峰值电平(dBFS)/过零率/
              直流偏置/动态范围/频谱质心(Hz)/带宽(近似)。
    """
    mono = data.mean(axis=1) if data.ndim > 1 else data
    n = len(mono)
    dur = n / sr

    rms = float(np.sqrt(np.mean(mono.astype(float) ** 2)))
    peak = float(np.max(np.abs(mono)))
    # 峰值电平 (dBFS)：相对满刻度 (1.0) 的对数
    peak_dbfs = 20 * np.log10(max(peak, 1e-9))
    # 过零率：相邻样本符号变化次数 / 样本数
    zcr = float(np.mean(np.abs(np.diff(np.signbit(mono)))))
    # 直流偏置 (归一化到 [-1,1])
    dc = float(np.mean(mono))
    # 动态范围 (dB)：峰值与 RMS 的比
    dyn_range = 20 * np.log10(max(peak, 1e-9) / max(rms, 1e-9))

    # 频谱质心 (Hz) 与带宽
    spec = np.abs(np.fft.rfft(mono * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    total = spec.sum()
    if total > 0:
        centroid = float((freqs * spec).sum() / total)
        bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spec).sum() / total))
    else:
        centroid = 0.0
        bandwidth = 0.0

    return {
        "采样率(Hz)": int(sr),
        "声道数": 1 if data.ndim == 1 else data.shape[1],
        "时长(s)": round(dur, 4),
        "样本数": int(n),
        "RMS": round(rms, 6),
        "峰值": round(peak, 6),
        "峰值电平(dBFS)": round(peak_dbfs, 2),
        "过零率": round(zcr, 6),
        "直流偏置": round(dc, 6),
        "动态范围(dB)": round(dyn_range, 2),
        "频谱质心(Hz)": round(centroid, 1),
        "带宽(Hz)": round(bandwidth, 1),
    }


def audio_bit_depth(path):
    """返回音频文件位深 (bits)；非 PCM 格式返回 None。"""
    try:
        with wave.open(path, "rb") as w:
            return w.getsampwidth() * 8
    except Exception:
        return None


# =========================================================================
# 录音工作线程 (避免阻塞 UI)
# =========================================================================
class Recorder(QThread):
    sig_level = Signal(float)        # 实时 RMS (dB)
    sig_finished = Signal(object, int, int, str)  # data, sr, channels, dev_name
    sig_error = Signal(str)

    def __init__(self, device, sr, channels, duration):
        super().__init__()
        self.device = device
        self.sr = sr
        self.channels = channels
        self.duration = duration
        self._frames = []
        self._running = False
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("stream status:", status)
        self._frames.append(indata.copy())
        if self._frames:
            rms = np.sqrt(np.mean(indata.astype(float) ** 2))
            db = 20 * np.log10(max(rms, 1e-9))
            self.sig_level.emit(db)

    def run(self):
        try:
            self._running = True
            self._stream = sd.InputStream(
                samplerate=self.sr, channels=self.channels, dtype="float32",
                device=self.device, callback=self._callback,
                blocksize=int(self.sr * 0.1),
            )
            self._stream.start()
            # 分段等待：定时模式下按 0.05s 步进检查 _running，使 stop() 能立即生效
            if self.duration and self.duration > 0:
                elapsed = 0.0
                while self._running and elapsed < self.duration:
                    time.sleep(0.05)
                    elapsed += 0.05
            else:
                # 手动模式：直到 stop() 被调用
                while self._running:
                    time.sleep(0.05)
            # 停止并关闭流
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        except Exception as e:
            self.sig_error.emit(str(e))
            return
        if self._frames:
            arr = np.concatenate(self._frames, axis=0)
            dev_name = "dev"
            try:
                dev_name = sd.query_devices(self.device)["name"]
            except Exception:
                pass
            self.sig_finished.emit(arr, self.sr, self.channels, dev_name)

    def stop(self):
        """请求停止录音：置 _running=False，并主动中止底层音频流。"""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


# =========================================================================
# Matplotlib 画布封装
# =========================================================================
def setup_mpl_font():
    if not HAS_MPL or matplotlib is None:
        return
    for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC",
              "Source Han Sans SC", "Arial Unicode MS", "DejaVu Sans"]:
        try:
            matplotlib.rcParams["font.sans-serif"] = [f]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            pass


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(10, 4)):
        self.fig = Figure(figsize=figsize, dpi=96)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        # 缩放/平移工具栏（提供 放大/缩小/平移/复位/保存 等按钮）
        self.toolbar = None
        if HAS_MPL:
            try:
                from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
                self.toolbar = NavigationToolbar2QT(self, parent)
            except Exception:
                self.toolbar = None

    def clear(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)


# =========================================================================
# 统一绘图后端：全局可切换 matplotlib / pyqtgraph
# =========================================================================
# 后端名称 -> 可用性
BACKENDS = {}
if HAS_MPL:
    BACKENDS["matplotlib"] = True
if HAS_PG:
    BACKENDS["pyqtgraph"] = True

# 全局后端开关，默认 matplotlib；可运行时切换
CURRENT_BACKEND = "matplotlib" if HAS_MPL else ("pyqtgraph" if HAS_PG else "none")


def set_backend(name):
    """切换全局绘图后端。返回是否成功。"""
    global CURRENT_BACKEND
    if name not in BACKENDS:
        return False
    CURRENT_BACKEND = name
    return True


# =========================================================================
# 统一绘图视图：根据 CURRENT_BACKEND 在 matplotlib / pyqtgraph 之间切换
# =========================================================================
class PlotView(QWidget):
    """统一的绘图视图封装。

    内部根据全局 CURRENT_BACKEND 持有 matplotlib 的 MplCanvas 或
    pyqtgraph 的 GraphicsLayoutWidget，对外暴露一致的高层绘图接口。
    切换后端后调用 rebuild() 即可重建底层控件。
    """

    def __init__(self, parent=None, figsize=(10, 4)):
        super().__init__(parent)
        self.figsize = figsize
        self._mpl = None          # matplotlib 画布
        self._pg = None           # pyqtgraph GraphicsLayoutWidget
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._build()

    # ---------- 底层构建 ----------
    def _build(self):
        """根据当前后端构建底层控件。"""
        # 清空旧控件
        for w in [self._mpl, self._pg]:
            if w is not None:
                self._layout.removeWidget(w)
                w.deleteLater()
        self._mpl = None
        self._pg = None

        if CURRENT_BACKEND == "pyqtgraph" and HAS_PG:
            self._pg = pg.GraphicsLayoutWidget()
            self._pg.setBackground("w")
            self._layout.addWidget(self._pg)
            self._subplots = []   # 记录当前子图 (PlotItem 列表)
        else:
            self._mpl = MplCanvas(self, figsize=self.figsize)
            self._layout.addWidget(self._mpl)
            if self._mpl.toolbar is not None:
                self._layout.addWidget(self._mpl.toolbar)
            self._subplots = []

    def rebuild(self):
        """切换后端后重建视图。"""
        self._build()

    @property
    def backend(self):
        return "pyqtgraph" if self._pg is not None else "matplotlib"

    # ---------- 统一高层接口 ----------
    def clear(self):
        """清空所有内容，返回一个用于绘制的主子图对象。"""
        self._subplots = []
        if self._pg is not None:
            self._pg.clear()
            p = self._pg.addPlot(row=0, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            self._subplots.append(p)
            return _PGSubplot(p)
        else:
            self._mpl.clear()
            return _MplSubplot(self._mpl.ax)

    def make_grid(self, n):
        """创建 n 个子图的网格布局，返回子图对象列表（支持多子图）。"""
        self._subplots = []
        if self._pg is not None:
            self._pg.clear()
            ncols = 2 if n > 1 else 1
            nrows = (n + ncols - 1) // ncols
            subs = []
            for i in range(n):
                r, c = divmod(i, ncols)
                p = self._pg.addPlot(row=r, col=c)
                p.showGrid(x=True, y=True, alpha=0.3)
                subs.append(_PGSubplot(p))
                self._subplots.append(p)
            return subs
        else:
            ncols = 2 if n > 1 else 1
            nrows = (n + ncols - 1) // ncols
            subs = []
            for i in range(n):
                ax = self._mpl.fig.add_subplot(nrows, ncols, i + 1)
                subs.append(_MplSubplot(ax))
            return subs

    def text_center(self, msg):
        """居中显示提示文字。"""
        if self._pg is not None:
            self._pg.clear()
            label = pg.LabelItem(msg)
            self._pg.addItem(label)
        else:
            ax = self._mpl.fig.add_subplot(111)
            ax.text(0.5, 0.5, msg, ha="center", va="center")
            ax.axis("off")

    def finalize(self):
        """绘制收尾：matplotlib 需 tight_layout + draw，pyqtgraph 无需。"""
        if self._pg is not None:
            return
        try:
            self._mpl.fig.tight_layout()
        except Exception:
            pass
        self._mpl.draw()


# ---------- 子图适配器：统一 matplotlib Axes 与 pyqtgraph PlotItem ----------
class _MplSubplot:
    """matplotlib Axes 的适配器。"""

    def __init__(self, ax):
        self.ax = ax

    def plot(self, x, y, color="#1f77b4", width=1, label=None, style="line"):
        if style == "log":
            self.ax.semilogy(x, y, color=color, linewidth=width, label=label)
        else:
            self.ax.plot(x, y, color=color, linewidth=width, label=label)

    def scatter(self, x, y, color="#2ca02c"):
        self.ax.plot(x, y, color=color, linewidth=1)

    def imshow(self, matrix, extent=None, cmap="magma", title=""):
        im = self.ax.imshow(matrix, aspect="auto", origin="lower",
                            extent=extent, cmap=cmap)
        self.ax.figure.colorbar(im, ax=self.ax, label="dB")

    def set_labels(self, title="", xlabel="", ylabel=""):
        if title:
            self.ax.set_title(title)
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)

    def grid(self, alpha=0.3):
        self.ax.grid(alpha=alpha)

    def legend(self, fontsize=8):
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=fontsize)

    def set_yticks(self, ticks=None):
        self.ax.set_yticks(ticks or [])

    def set_axis_off(self):
        self.ax.axis("off")

    def text(self, msg):
        self.ax.text(0.5, 0.5, msg, ha="center", va="center")
        self.ax.axis("off")


class _PGSubplot:
    """pyqtgraph PlotItem 的适配器。"""

    def __init__(self, plot_item):
        self.p = plot_item
        self._legend = None
        self._legend_entries = []

    def plot(self, x, y, color="#1f77b4", width=1, label=None, style="line"):
        pen = pg.mkPen(color, width=width)
        if style == "log":
            self.p.setLogMode(x=False, y=True)
        self.p.plot(x, y, pen=pen, name=label)

    def scatter(self, x, y, color="#2ca02c"):
        self.p.plot(x, y, pen=None, symbol="o", symbolSize=3,
                    symbolBrush=pg.mkBrush(color))

    def imshow(self, matrix, extent=None, cmap="magma", title=""):
        img = pg.ImageItem(matrix)
        if extent:
            x0, x1, y0, y1 = extent
            img.setRect(x0, y0, x1 - x0, y1 - y0)
        img.setColorMap(pg.colormap.get(cmap))
        self.p.addItem(img)

    def set_labels(self, title="", xlabel="", ylabel=""):
        if title:
            self.p.setTitle(title)
        if xlabel:
            self.p.setLabel("bottom", xlabel)
        if ylabel:
            self.p.setLabel("left", ylabel)

    def grid(self, alpha=0.3):
        self.p.showGrid(x=True, y=True, alpha=alpha)

    def legend(self, fontsize=8):
        # pyqtgraph 用 addLegend
        try:
            if not self.p.legend:
                self.p.addLegend()
        except Exception:
            pass

    def set_yticks(self, ticks=None):
        self.p.getAxis("left").setTicks([])

    def set_axis_off(self):
        self.p.hideAxis("left")
        self.p.hideAxis("bottom")

    def text(self, msg):
        self.p.setTitle(msg)


# =========================================================================
# 子窗口 (放大查看单个图 / 显示指标)
# =========================================================================
class PlotDialog(QDialog):
    """独立子窗口：放大查看单个图（遵循全局绘图后端设置）。"""

    def __init__(self, parent=None, title="图"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(960, 640)
        v = QVBoxLayout(self)
        self.view = PlotView(self, figsize=(9, 5))
        v.addWidget(self.view)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        v.addWidget(btn)


class MetricsDialog(QDialog):
    """独立子窗口：完整显示音频性能指标（占空间小，点击按钮才弹出）。"""

    def __init__(self, parent=None, title="音频性能指标"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 500)
        v = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["指标", "值"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        v.addWidget(self.table)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        v.addWidget(btn)

    def set_metrics(self, metrics):
        """填充指标表格。"""
        self.table.setRowCount(len(metrics))
        for r, (k, val) in enumerate(metrics.items()):
            self.table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.table.setItem(r, 1, QTableWidgetItem(str(val)))


# =========================================================================
# 录音主窗口
# =========================================================================
class MainWindow(QMainWindow):
    """主窗口：UI 布局来自 main_window.ui (Qt Designer 设计)，
       本类只负责业务逻辑与信号连接，便于用 Designer 单独改界面。"""

    def __init__(self):
        super().__init__()
        if not HAS_QT:
            raise RuntimeError(f"PySide6 未安装: {_qt_err}")
        if sd is None:
            QMessageBox.critical(None, "错误", f"sounddevice 未安装: {_sd_err}")
        setup_mpl_font()
        # 加载 .ui 文件：把 main_window.ui 中的 centralwidget / 控件挂到 self
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_window.ui")
        _load_ui(ui_path, self)
        # QUiLoader 不会自动把子控件暴露为 self.xxx 属性，
        # 这里统一按 objectName 绑定，使 self.device_cb 等写法可用（也方便 Designer 增删控件）。
        self._bind_ui_widgets(self)
        self.resize(1100, 760)
        self.recorder = None
        self.last_data = None
        self.last_meta = None
        self.save_dir = self._default_save_dir()
        self._player = None
        self._audio_out = None       # 保持播放器输出设备引用，避免 GC
        self._preview_path = None    # 试听临时文件路径
        self.vw_file = None          # 查看器当前打开的文件路径
        self.vw_metrics = None       # 查看器当前计算的指标
        self._an_metrics = None      # 分析页当前指标（供弹窗）
        self._an_subplots = None     # 分析页当前子图信息（供放大查看）
        self._cmp_metrics_data = None  # 对比页指标对比数据（供弹窗）

        # 将统一绘图视图嵌入 .ui 预留的占位控件（支持 matplotlib / pyqtgraph 切换）
        self.canvas = PlotView(self, figsize=(10, 3.5))
        self._replace_placeholder("canvas_placeholder", self.canvas)
        self.an_canvas = PlotView(self, figsize=(10, 6))
        self._replace_placeholder("an_canvas_placeholder", self.an_canvas)
        self.cmp_canvas = PlotView(self, figsize=(11, 6))
        self._replace_placeholder("cmp_canvas_placeholder", self.cmp_canvas)
        self.vw_canvas = PlotView(self, figsize=(11, 5))
        self._replace_placeholder("vw_canvas_placeholder", self.vw_canvas)
        # 保存所有 PlotView，供后端切换时统一 rebuild
        self._plot_views = [self.canvas, self.an_canvas, self.cmp_canvas, self.vw_canvas]

        # 分析方法勾选框映射（对应 .ui 中的 chk_* 控件）
        # 每个方法对应一个绘图函数，勾选后运行分析时绘制对应子图
        self.an_checks = {
            "时域波形": (self.chk_time, self._plot_time),
            "频谱": (self.chk_freq, self._plot_freq),
            "语谱图": (self.chk_spec, self._plot_specgram),
            "基频": (self.chk_pitch, self._plot_pitch),
            "窗函数对比": (self.chk_window, self._plot_window_compare),
            "降噪对比": (self.chk_denoise, self._plot_denoise),
        }

        self._connect_signals()
        # 后端选择下拉框初始化为当前后端
        self.backend_cb.setCurrentText(CURRENT_BACKEND)
        # 分析/对比目录默认指向上次录音输出目录
        self.an_dir_le.setText(self.save_dir)
        self.cmp_dir_le.setText(self.save_dir)
        # 保存目录显示
        self.save_dir_le.setText(self.save_dir)
        self.refresh_devices()

    # ---------- UI 辅助 ----------
    def _bind_ui_widgets(self, root):
        """把 root 下所有带 objectName 的子控件按名字绑定到 self，
        使 self.<objectName> 可直接访问（QUiLoader 默认不自动绑定）。"""
        for w in root.findChildren(QWidget):
            name = w.objectName()
            if name and not hasattr(self, name):
                setattr(self, name, w)

    def _replace_placeholder(self, ph_name, widget):
        """把 .ui 中名为 ph_name 的占位 QWidget 替换为真实 widget (如 Matplotlib 画布)。

        占位控件可能嵌套在 Tab 内的布局中，故用 findChild 递归查找，
        先记录其位置，再 removeWidget + insertWidget 替换（比 replaceWidget 更可靠）。
        """
        ph = self.findChild(QWidget, ph_name)
        if ph is None:
            return
        parent_layout = ph.parentWidget().layout()
        if parent_layout is None:
            return
        idx = -1
        for i in range(parent_layout.count()):
            item = parent_layout.itemAt(i)
            if item and item.widget() is ph:
                idx = i
                break
        if idx < 0:
            return
        parent_layout.removeWidget(ph)
        ph.deleteLater()
        parent_layout.insertWidget(idx, widget)
        # 若该控件带缩放工具栏，则插入到画布下方（下一个位置）
        tb = getattr(widget, "toolbar", None)
        if tb is not None:
            parent_layout.insertWidget(idx + 1, tb)
        # 同步更新 self 上的占位引用（旧占位名 -> 新控件）
        if hasattr(self, ph_name):
            setattr(self, ph_name, widget)

    def _connect_signals(self):
        """连接 .ui 中各控件的信号到业务逻辑方法。"""
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.rec_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.save_btn.clicked.connect(self.save_recording)
        self.play_btn.clicked.connect(self.play_last)
        self.choose_dir_btn.clicked.connect(self.choose_save_dir)
        self.open_dir_btn.clicked.connect(self.open_save_dir)
        self.dur_cb.currentTextChanged.connect(
            lambda t: self.custom_sb.setEnabled(t == "自定义"))
        self.an_browse_btn.clicked.connect(self._browse_an_dir)
        self.an_scan_btn.clicked.connect(self.scan_analysis)
        self.an_run_btn.clicked.connect(self.run_analysis)
        self.an_open_btn.clicked.connect(self.open_analysis_file)
        self.an_list.itemSelectionChanged.connect(self._on_an_selection)
        self.an_list.itemDoubleClicked.connect(self._on_an_double_clicked)
        self.an_metrics_btn.clicked.connect(self.show_analysis_metrics)
        self.an_zoom_btn.clicked.connect(self.zoom_subplot)
        self.cmp_browse_btn.clicked.connect(self._browse_cmp_dir)
        self.cmp_scan_btn.clicked.connect(self.scan_compare)
        self.cmp_run_btn.clicked.connect(self.run_compare)
        self.cmp_open_btn.clicked.connect(self.open_compare_file)
        self.cmp_list.itemSelectionChanged.connect(self._on_cmp_selection)
        self.cmp_metrics_btn.clicked.connect(self.show_cmp_metrics)
        self.vw_open_btn.clicked.connect(self.open_viewer_file)
        self.vw_export_btn.clicked.connect(self.export_viewer_result)
        self.vw_refresh_btn.clicked.connect(self.refresh_viewer_plot)
        self.vw_win_cb.currentIndexChanged.connect(self.refresh_viewer_plot)
        self.vw_fft_cb.currentIndexChanged.connect(self.refresh_viewer_plot)
        self.vw_mode_cb.currentIndexChanged.connect(self.refresh_viewer_plot)
        self.vw_db_chk.toggled.connect(self.refresh_viewer_plot)
        self.apply_backend_btn.clicked.connect(self.switch_backend)

    # ---------- 工具 ----------
    def _default_save_dir(self):
        d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "recordings"))
        os.makedirs(d, exist_ok=True)
        return d

    def choose_save_dir(self):
        """选择录音保存目录。"""
        p = QFileDialog.getExistingDirectory(self, "选择保存目录", self.save_dir)
        if p:
            self.save_dir = p
            self.save_dir_le.setText(p)
            self._set_status(f"保存目录已设为: {p}")

    def open_save_dir(self):
        """在系统文件管理器中打开保存目录。"""
        d = self.save_dir
        os.makedirs(d, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)  # Windows 资源管理器
            elif sys.platform == "darwin":
                import subprocess; subprocess.Popen(["open", d])
            else:
                import subprocess; subprocess.Popen(["xdg-open", d])
            self._set_status(f"已打开目录: {d}")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"无法打开目录: {e}")

    def _set_status(self, msg):
        self.status.setText(msg)

    def switch_backend(self):
        """切换全局绘图引擎（matplotlib / pyqtgraph）并重建所有画布。"""
        name = self.backend_cb.currentText()
        if name not in BACKENDS:
            QMessageBox.warning(self, "提示", f"后端 {name} 不可用（未安装）")
            return
        if not set_backend(name):
            QMessageBox.warning(self, "提示", f"无法切换到后端 {name}")
            return
        # 重建所有 PlotView
        for pv in self._plot_views:
            pv.rebuild()
        self._set_status(f"绘图引擎已切换为: {name}")
        QMessageBox.information(self, "切换成功",
                                f"绘图引擎已切换为 {name}。\n"
                                "当前已绘制的图需要重新运行相应操作才会以新引擎渲染。")

    # ---------- 设备 ----------
    def refresh_devices(self):
        try:
            all_d, ins, default = list_audio_devices()
        except Exception as e:
            QMessageBox.critical(self, "设备扫描失败", str(e))
            return
        self.device_cb.clear()
        for d in ins:
            label = f"[{d['index']}] {d['name']}  (ch={d['max_in']}, sr={d['default_sr']:.0f}, {d['hostapi']})"
            self.device_cb.addItem(label, d["index"])
        if default is not None:
            idx = self.device_cb.findData(int(default))
            if idx >= 0:
                self.device_cb.setCurrentIndex(idx)
        # 同时刷新输出设备（用于试听）
        self.refresh_output_devices()
        self._set_status(f"发现 {len(ins)} 个输入设备")

    def refresh_output_devices(self):
        """刷新试听输出设备下拉框。"""
        try:
            outs, default_out = list_output_devices()
        except Exception:
            return
        self.out_dev_cb.clear()
        for d in outs:
            label = f"[{d['index']}] {d['name']}  (ch={d['max_out']})"
            self.out_dev_cb.addItem(label, d["index"])
        if default_out is not None:
            idx = self.out_dev_cb.findData(int(default_out))
            if idx >= 0:
                self.out_dev_cb.setCurrentIndex(idx)

    # ---------- 录音 ----------
    def start_recording(self):
        if self.recorder and self.recorder.isRunning():
            return
        if sd is None:
            QMessageBox.critical(self, "错误", "sounddevice 未安装")
            return
        dev = self.device_cb.currentData()
        if dev is None:
            QMessageBox.warning(self, "提示", "请先选择输入设备")
            return
        sr = int(self.sr_cb.currentText())
        ch = int(self.channels_cb.currentText())
        txt = self.dur_cb.currentText()
        dur_map = {"5 秒":5,"10 秒":10,"30 秒":30,"1 分钟":60,"3 分钟":180}
        dur = dur_map.get(txt, -1)
        if dur == -1:
            dur = self.custom_sb.value()
        self.recorder = Recorder(dev, sr, ch, dur)
        self.recorder.sig_level.connect(self._on_level)
        self.recorder.sig_finished.connect(self._on_finished)
        self.recorder.sig_error.connect(lambda e: QMessageBox.critical(self, "录音失败", e))
        self.recorder.start()
        self.rec_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False); self.play_btn.setEnabled(False)
        self._set_status(f"⏺ 录音中... 目标 {dur}s")
        if dur > 0:
            self.progress.setRange(0, dur*20); self.progress.setValue(0)
            self._timer = QTimer(); self._timer.timeout.connect(self._tick); self._timer.start(50)
        else:
            self.progress.setRange(0, 0)

    def _tick(self):
        if self.progress.maximum() > 0:
            v = self.progress.value() + 1
            self.progress.setValue(min(v, self.progress.maximum()))
            self._set_status(f"⏺ 录音中... {v/20:.1f}s")

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self._set_status("正在停止...")

    def _on_level(self, db):
        if self.progress.maximum() == 0:  # 手动模式
            self._set_status(f"⏺ 录音中... 电平 {db:.1f} dB")

    def _on_finished(self, data, sr, ch, dev_name):
        self.last_data = data
        self.last_meta = {"sr": sr, "ch": ch, "dev": dev_name}
        self.rec_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True); self.play_btn.setEnabled(True)
        dur = len(data) / sr
        peak = float(np.max(np.abs(data)))
        self._set_status(f"✔ 完成: {dur:.2f}s  sr={sr} ch={ch} peak={peak:.3f}  设备={dev_name}")
        self.progress.setRange(0, 1); self.progress.setValue(1)
        if hasattr(self, "_timer"):
            self._timer.stop()
        self._draw_waveform(data, sr)
        # 试听：保存临时文件，并准备播放器（使用选定的输出设备）
        try:
            mono = data.mean(axis=1) if data.ndim > 1 else data
            self._preview_path = os.path.join(self.save_dir, "_preview.wav")
            sf.write(self._preview_path, mono, sr)
            self._init_player()
            self.player_label.setText(f"试听：就绪（输出:{self.out_dev_cb.currentText()}）")
        except Exception as e:
            self.player_label.setText(f"试听不可用: {e}")

    def _init_player(self):
        """初始化/更新播放器，绑定选定的输出设备。"""
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
        from PySide6.QtCore import QUrl
        out_idx = self.out_dev_cb.currentData()
        audio_out = None
        if out_idx is not None:
            # 用 sounddevice 的索引映射到 Qt 输出设备
            try:
                outs, _ = list_output_devices()
                sd_name = next((d["name"] for d in outs if d["index"] == out_idx), None)
                if sd_name:
                    for dev in QMediaDevices.audioOutputs():
                        if sd_name in dev.description() or dev.description() in sd_name:
                            audio_out = QAudioOutput(dev)
                            break
            except Exception:
                pass
        if audio_out is None:
            audio_out = QAudioOutput()  # 使用系统默认输出
        # 保持 audio_out 引用，避免被 GC 回收导致无声
        self._audio_out = audio_out
        if self._player is None:
            self._player = QMediaPlayer()
        self._player.setAudioOutput(audio_out)
        if getattr(self, "_preview_path", None):
            self._player.setSource(QUrl.fromLocalFile(self._preview_path))

    def _draw_waveform(self, data, sr):
        mono = data.mean(axis=1) if data.ndim > 1 else data
        t = np.arange(len(mono)) / sr
        sub = self.canvas.clear()
        sub.plot(t, mono, color="#1f77b4", width=0.8)
        sub.set_labels("最近录音波形", "时间 (s)", "幅度")
        sub.grid()
        self.canvas.finalize()

    def play_last(self):
        if self._player:
            # 每次播放前重新绑定输出设备（用户可能切换了）
            try:
                self._init_player()
            except Exception as e:
                self.player_label.setText(f"试听不可用: {e}")
                return
            self._player.play()
            self.player_label.setText(f"试听播放中...（输出:{self.out_dev_cb.currentText()}）")

    def save_recording(self):
        if self.last_data is None:
            QMessageBox.warning(self, "提示", "没有可保存的录音")
            return
        fmt_map = {"WAV (无损,推荐)":"wav","FLAC (无损压缩)":"flac","OGG/Vorbis":"ogg","MP3 (需 lameenc)":"mp3"}
        fmt = fmt_map[self.fmt_cb.currentText()]
        name = self.name_le.text().strip()
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name:
            name = f"rec_{ts}"
        if not name.lower().endswith("." + fmt):
            name = f"{name}.{fmt}"
        path = os.path.join(self.save_dir, name)
        try:
            save_audio(self.last_data, self.last_meta["sr"], path, fmt)
            sz = os.path.getsize(path) / 1024
            self._set_status(f"💾 已保存: {path} ({sz:.1f} KB)")
            QMessageBox.information(self, "保存成功", path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    # ---------- 分析 ----------
    def _browse_an_dir(self):
        p = QFileDialog.getExistingDirectory(self, "选择目录", self.an_dir_le.text())
        if p:
            self.an_dir_le.setText(p)

    def scan_analysis(self):
        d = self.an_dir_le.text().strip()
        self.an_list.clear()
        for f in self._collect_files(d):
            self._add_file_item(self.an_list, f)
        self._set_status(f"扫描到 {self.an_list.count()} 个音频文件")

    def _collect_files(self, d):
        """扫描目录/文件，返回完整路径列表（修复：原来返回纯文件名导致读取失败）。"""
        if os.path.isfile(d):
            return [d] if d.lower().endswith(AUDIO_EXTS) else []
        if os.path.isdir(d):
            return sorted(
                os.path.join(d, p) for p in os.listdir(d)
                if os.path.join(d, p).lower().endswith(AUDIO_EXTS)
            )
        return []

    @staticmethod
    def _add_file_item(list_widget, path):
        """向列表添加文件项：显示文件名，用 item data 保存完整路径。"""
        from PySide6.QtWidgets import QListWidgetItem
        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.UserRole, path)          # 完整路径
        item.setToolTip(path)                     # 悬停显示完整路径
        list_widget.addItem(item)

    @staticmethod
    def _selected_paths(list_widget):
        """返回列表中选中项（或全部项）的完整路径列表。"""
        items = list_widget.selectedItems()
        if not items:
            items = [list_widget.item(i) for i in range(list_widget.count())]
        return [it.data(Qt.UserRole) for it in items]

    def open_analysis_file(self):
        """直接打开单个音频文件，立即加载并实时显示指标 + 运行分析。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开音频文件", self.save_dir,
            "音频文件 (*.wav *.flac *.ogg *.mp3 *.m4a *.aac);;所有文件 (*.*)")
        if not path:
            return
        self.an_dir_le.setText(path)
        # 用单个文件直接分析
        self._analyze_file(path)

    def _on_an_selection(self):
        """列表选择变化时，高亮显示当前选中的文件名。"""
        sel = self.an_list.selectedItems()
        if sel:
            self.an_cur_label.setText(f"当前文件: {sel[0].text()}")
        else:
            self.an_cur_label.setText("当前文件: 未选择")

    def _on_an_double_clicked(self, item):
        """双击列表项：立即替换为当前文件并运行分析。"""
        path = item.data(Qt.UserRole)
        if path:
            self._analyze_file(path)

    def run_analysis(self):
        files = self._selected_paths(self.an_list)
        if not files:
            QMessageBox.warning(self, "提示", "请先扫描/打开并选择文件")
            return
        self._analyze_file(files[0])

    def _analyze_file(self, path):
        """加载文件，实时显示指标，并按勾选的分析方法绘制对应子图。"""
        try:
            data, mono, sr = load_audio(path)
        except Exception as e:
            QMessageBox.critical(self, "分析失败", f"无法读取 {path}:\n{e}")
            return

        # 1) 更新"当前文件"标签
        self.an_cur_label.setText(f"当前文件: {os.path.basename(path)}")

        # 2) 实时显示性能指标
        metrics = compute_audio_metrics(data, sr)
        self._fill_analysis_metrics(metrics)

        # 3) 按勾选方法绘制子图（每个方法一个子图，方法名与图一一对应）
        chosen = [name for name, (cb, _fn) in self.an_checks.items() if cb.isChecked()]
        fname = os.path.basename(path)
        if not chosen:
            self.an_canvas.text_center("请至少勾选一种分析方法")
            self.an_canvas.finalize()
            self._set_status(f"已加载: {fname} (未选择分析方法)")
            return

        # 创建子图网格，每个方法一个子图
        subs = self.an_canvas.make_grid(len(chosen))
        # 记录每个子图的绘制信息，供“放大查看”子窗口使用
        self._an_subplots = []
        for i, name in enumerate(chosen):
            plot_fn = self.an_checks[name][1]
            plot_fn(subs[i], mono, sr, fname)
            self._an_subplots.append((name, plot_fn, mono, sr, fname))
        # 填充“放大查看子图”下拉框
        self.an_sub_cb.clear()
        for name, *_ in self._an_subplots:
            self.an_sub_cb.addItem(name)

        self.an_canvas.finalize()
        self._set_status(f"分析完成: {fname}  (方法: {','.join(chosen)})")

    def _fill_analysis_metrics(self, metrics):
        """保存指标到 self._an_metrics，供“查看指标”按钮弹子窗口显示。"""
        self._an_metrics = metrics

    def show_analysis_metrics(self):
        """弹子窗口显示当前文件的性能指标。"""
        metrics = getattr(self, "_an_metrics", None)
        if not metrics:
            QMessageBox.warning(self, "提示", "请先运行分析")
            return
        dlg = MetricsDialog(self, title="音频性能指标")
        dlg.set_metrics(metrics)
        dlg.exec()

    def zoom_subplot(self):
        """把当前选中的子图放大到独立子窗口查看（pyqtgraph 优先）。"""
        idx = self.an_sub_cb.currentIndex()
        if idx < 0 or not getattr(self, "_an_subplots", None):
            QMessageBox.warning(self, "提示", "请先运行分析并选择子图")
            return
        name, plot_fn, mono, sr, fname = self._an_subplots[idx]
        dlg = PlotDialog(self, title=f"放大查看 - {name}")
        sub = dlg.view.clear()
        plot_fn(sub, mono, sr, fname)
        dlg.view.finalize()
        dlg.exec()

    # ---------- 分析方法绘图（每个方法对应一个子图，sub 为统一适配器）----------
    def _plot_time(self, sub, mono, sr, name):
        t = np.arange(len(mono)) / sr
        sub.plot(t, mono, color="#1f77b4", width=0.8)
        sub.set_labels(f"时域波形 - {name}", "时间 (s)", "幅度")
        sub.grid()

    def _plot_freq(self, sub, mono, sr, name):
        N = len(mono)
        spec = np.abs(np.fft.rfft(mono * np.hanning(N)))
        freqs = np.fft.rfftfreq(N, 1/sr)
        sub.plot(freqs, np.maximum(spec, 1e-12), color="#d62728", style="log")
        sub.set_labels("频谱 (FFT)", "频率 (Hz)", "幅度")
        sub.grid()

    def _plot_specgram(self, sub, mono, sr, name):
        """语谱图 (STFT)。（分析页专用，避免与查看器 _plot_spectrogram 重名）"""
        nfft = 1024
        hop = nfft // 2
        n_frames = max(1, (len(mono) - nfft) // hop + 1)
        if len(mono) < nfft:
            sub.text("音频过短，无法绘制语谱图")
            return
        frames = []
        for i in range(n_frames):
            seg = mono[i * hop: i * hop + nfft]
            frames.append(np.abs(np.fft.rfft(seg * np.hanning(nfft))))
        S = np.array(frames).T
        S_db = 20 * np.log10(np.maximum(S, 1e-12))
        sub.imshow(S_db, extent=[0, len(mono) / sr, 0, sr / 2], cmap="magma")
        sub.set_labels("语谱图 (STFT)", "时间 (s)", "频率 (Hz)")

    def _plot_pitch(self, sub, mono, sr, name):
        """基频 (f0) 轨迹：用自相关法逐帧估计。"""
        frame = int(sr * 0.04)  # 40ms 帧
        hop = frame // 2
        if len(mono) < frame:
            sub.text("音频过短，无法估计基频")
            return
        f0s, times = [], []
        for i in range(0, len(mono) - frame, hop):
            seg = mono[i: i + frame]
            seg = seg - seg.mean()
            ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
            lo = int(sr / 500); hi = int(sr / 50)
            hi = min(hi, len(ac) - 1)
            window = ac[lo + 1: hi]
            if lo >= hi or window.size == 0:
                continue
            peak_val = float(np.max(window))
            if peak_val <= 0:
                continue
            lag = lo + 1 + int(np.argmax(window))
            f0s.append(sr / lag)
            times.append((i + frame / 2) / sr)
        if not f0s:
            sub.text("未检测到明显基频")
            return
        sub.plot(times, f0s, color="#2ca02c", width=1.5)
        sub.set_labels("基频 (f0) 轨迹", "时间 (s)", "频率 (Hz)")
        sub.grid()

    def _plot_window_compare(self, sub, mono, sr, name):
        """窗函数对比：同一段信号加不同窗后的频谱对比。"""
        N = min(len(mono), 4096)
        seg = mono[:N]
        freqs = np.fft.rfftfreq(N, 1 / sr)
        windows = {"汉宁": np.hanning, "汉明": np.hamming,
                   "布莱克曼": np.blackman, "矩形": np.ones}
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
        for (wname, wfn), color in zip(windows.items(), colors):
            spec = np.abs(np.fft.rfft(seg * wfn(N)))
            sub.plot(freqs, np.maximum(spec, 1e-12), color=color, width=1, label=wname, style="log")
        sub.set_labels("窗函数频谱对比", "频率 (Hz)", "幅度")
        sub.legend(fontsize=7); sub.grid()

    def _plot_denoise(self, sub, mono, sr, name):
        """降噪对比：原始 vs 简单谱减法降噪后的波形。"""
        N = len(mono)
        noise = mono[: max(1, N // 10)]
        noise_spec = np.abs(np.fft.rfft(noise * np.hanning(len(noise))))
        spec = np.fft.rfft(mono * np.hanning(N))
        gain = np.maximum(1 - np.mean(noise_spec) / (np.abs(spec) + 1e-12), 0)
        denoised = np.fft.irfft(spec * gain, n=N)
        t = np.arange(N) / sr
        sub.plot(t, mono, color="#999999", width=0.8, label="原始")
        sub.plot(t, denoised, color="#1f77b4", width=1.0, label="降噪后")
        sub.set_labels("降噪对比 (谱减法)", "时间 (s)", "幅度")
        sub.legend(fontsize=7); sub.grid()

    # ---------- 对比 ----------
    def _browse_cmp_dir(self):
        p = QFileDialog.getExistingDirectory(self, "选择目录", self.cmp_dir_le.text())
        if p:
            self.cmp_dir_le.setText(p)

    def scan_compare(self):
        d = self.cmp_dir_le.text().strip()
        self.cmp_list.clear()
        for f in self._collect_files(d):
            self._add_file_item(self.cmp_list, f)
        self._update_cmp_label()
        self._set_status(f"对比扫描到 {self.cmp_list.count()} 个文件")

    def open_compare_file(self):
        """直接打开音频文件并加入对比列表。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开音频文件", self.save_dir,
            "音频文件 (*.wav *.flac *.ogg *.mp3 *.m4a *.aac);;所有文件 (*.*)")
        if not path:
            return
        self._add_file_item(self.cmp_list, path)
        self._set_status(f"已加入对比: {os.path.basename(path)}")
        self._update_cmp_label()

    def _on_cmp_selection(self):
        """更新已选文件数量标签。"""
        self._update_cmp_label()

    def _update_cmp_label(self):
        n = len(self.cmp_list.selectedItems())
        total = self.cmp_list.count()
        if n == 0:
            self.cmp_cur_label.setText(f"已选文件: 0 个 (点击/双击列表项选择, 共 {total} 个)")
        else:
            self.cmp_cur_label.setText(f"已选文件: {n} 个 / 共 {total} 个")

    def run_compare(self):
        files = self._selected_paths(self.cmp_list)
        if len(files) < 2:
            QMessageBox.warning(self, "提示", "对比需要至少 2 个文件")
            return
        if not HAS_MPL or plt is None:
            QMessageBox.warning(self, "提示", "matplotlib 未安装，无法绘图")
            return
        mode = self.cmp_mode_cb.currentText()
        try:
            if mode == "波形叠加对比":
                self._compare_waveform(files)
            elif mode == "频谱对比":
                self._compare_spectrum(files)
            elif mode == "指标对比表":
                self._compare_metrics_table(files)
            else:  # 相似度矩阵
                self._compare_similarity(files)
            self._set_status(f"对比完成: {len(files)} 个文件 ({mode})")
        except Exception as e:
            import traceback
            detail = traceback.format_exc()
            QMessageBox.critical(
                self, "对比失败",
                f"对比方式: {mode}\n文件: {os.path.basename(files[0]) if files else '无'}\n"
                f"错误: {e}\n\n详细:\n{detail[-800:]}")

    def _load_all(self, files, target_sr=None):
        """批量加载音频，返回 (mono_list, sr_list, name_list)。
        可选统一重采样到 target_sr，避免不同采样率导致时间轴错位。"""
        monos, srs, names = [], [], []
        for f in files:
            data, mono, sr = load_audio(f)
            if target_sr is not None and sr != target_sr:
                from scipy.signal import resample
                n = int(len(mono) * target_sr / sr)
                mono = resample(mono.astype(float), n)
                sr = target_sr
            monos.append(mono)
            srs.append(sr)
            names.append(os.path.basename(f))
        return monos, srs, names

    def _compare_waveform(self, files):
        """波形叠加对比：统一采样率后归一化+偏移叠加。"""
        _, sr0, _ = load_audio(files[0])
        monos, _, names = self._load_all(files, target_sr=sr0)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        sub = self.cmp_canvas.clear()
        for i, m in enumerate(monos):
            m = m / (np.max(np.abs(m)) + 1e-12) * 0.9
            t = np.arange(len(m)) / sr0
            sub.plot(t, m + i * 1.2, color=colors[i % 10], width=0.6, label=names[i])
        sub.set_labels("波形叠加对比 (归一化+偏移, 统一采样率)", "时间 (s)", "")
        sub.set_yticks([])
        sub.legend(fontsize=8); sub.grid()
        self.cmp_canvas.finalize()

    def _compare_spectrum(self, files):
        """频谱对比：多文件 FFT 频谱叠加。"""
        _, sr0, _ = load_audio(files[0])
        monos, _, names = self._load_all(files, target_sr=sr0)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        sub = self.cmp_canvas.clear()
        nfft = 4096
        for i, m in enumerate(monos):
            seg = m if len(m) < nfft else m[:nfft]
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            freqs = np.fft.rfftfreq(len(seg), 1 / sr0)
            sub.plot(freqs, np.maximum(spec, 1e-12), color=colors[i % 10], width=1, label=names[i], style="log")
        sub.set_labels("频谱对比", "频率 (Hz)", "幅度")
        sub.legend(fontsize=8); sub.grid()
        self.cmp_canvas.finalize()

    def _compare_metrics_table(self, files):
        """指标对比表：计算并保存指标，弹子窗口展示。"""
        headers = ["指标"]
        metrics_list = []
        for f in files:
            data, mono, sr = load_audio(f)
            metrics_list.append(compute_audio_metrics(data, sr))
            headers.append(os.path.basename(f))
        # 保存数据供“查看指标对比”按钮弹窗
        self._cmp_metrics_data = (headers, metrics_list)
        # 画布上给提示
        self.cmp_canvas.text_center("指标对比已计算\n点击“查看指标对比”按钮查看")
        self.cmp_canvas.finalize()
        self._set_status("指标对比已计算，点击“查看指标对比”按钮查看详情")

    def show_cmp_metrics(self):
        """弹子窗口显示多文件指标对比表。"""
        data = getattr(self, "_cmp_metrics_data", None)
        if not data:
            QMessageBox.warning(self, "提示", "请先运行“指标对比表”分析")
            return
        headers, metrics_list = data
        dlg = QDialog(self)
        dlg.setWindowTitle("多文件指标对比")
        dlg.resize(760, 520)
        v = QVBoxLayout(dlg)
        t = QTableWidget()
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.NoSelection)
        keys = list(metrics_list[0].keys())
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(keys))
        for r, k in enumerate(keys):
            t.setItem(r, 0, QTableWidgetItem(str(k)))
            for c, m in enumerate(metrics_list):
                t.setItem(r, c + 1, QTableWidgetItem(str(m[k])))
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, len(headers)):
            hh.setSectionResizeMode(c, QHeaderView.Stretch)
        v.addWidget(t)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(dlg.reject)
        v.addWidget(btn)
        dlg.exec()

    def _compare_similarity(self, files):
        """相似度矩阵：两两文件间的相关系数热力图。"""
        _, sr0, _ = load_audio(files[0])
        monos, _, names = self._load_all(files, target_sr=sr0)
        n = len(monos)
        # 截断到最短长度，保证可计算
        min_len = min(len(m) for m in monos)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a = np.asarray(monos[i][:min_len], dtype=float)
                b = np.asarray(monos[j][:min_len], dtype=float)
                if float(np.std(a)) > 1e-9 and float(np.std(b)) > 1e-9:
                    mat[i, j] = float(np.corrcoef(a, b)[0, 1])
                else:
                    mat[i, j] = 0.0
        sub = self.cmp_canvas.clear()
        sub.imshow(mat, extent=[-0.5, n - 0.5, -0.5, n - 0.5], cmap="viridis")
        sub.set_labels("相似度矩阵 (波形相关系数)", "", "")
        # matplotlib 下标注数值和刻度；pyqtgraph 下用图像即可
        if self.cmp_canvas.backend == "matplotlib":
            ax = sub.ax
            ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=8)
        self.cmp_canvas.finalize()

    # ---------- 音频查看器 ----------
    def open_viewer_file(self):
        """打开单个音频文件，展示其信息、指标、波形与频谱。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", self.save_dir,
            "音频文件 (*.wav *.flac *.ogg *.mp3 *.m4a *.aac);;所有文件 (*.*)")
        if not path:
            return
        try:
            data, mono, sr = load_audio(path)
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法读取音频文件:\n{e}")
            return

        self.vw_file = path
        self.vw_file_le.setText(path)
        # 保存原始数据，供参数调整后实时重绘
        self.vw_data = data
        self.vw_mono = mono
        self.vw_sr = sr

        # 1) 基本信息
        fmt = os.path.splitext(path)[1].lstrip(".").upper()
        bit_depth = audio_bit_depth(path)
        self.vw_basic = {
            "文件": os.path.basename(path),
            "路径": path,
            "格式": fmt,
            "位深": f"{bit_depth} bit" if bit_depth else "非 PCM (压缩格式)",
            "文件大小": f"{os.path.getsize(path) / 1024:.1f} KB",
        }

        # 2) 指标
        self.vw_metrics = compute_audio_metrics(data, sr)

        # 3) 填充到可勾选表格（支持显示/隐藏）
        self._fill_viewer_table()

        # 4) 波形 + 频谱
        self._draw_viewer_plot()
        self._set_status(f"已打开: {os.path.basename(path)}  (时长 {self.vw_metrics['时长(s)']}s)")

    def _fill_viewer_table(self):
        """把基本信息与指标填充到 QTableWidget，指标行带勾选框（可显示/隐藏）。"""
        table = self.vw_info_table
        table.clear()
        # 3 列：勾选 | 指标名 | 值
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["显示", "项目", "值"])
        # 基本信息行（无勾选框）
        basic = self.vw_basic
        metrics = self.vw_metrics
        rows = list(basic.items()) + list(metrics.items())
        table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            is_metric = k in metrics
            if is_metric:
                # 指标行：可勾选
                chk_item = QTableWidgetItem()
                chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                chk_item.setCheckState(Qt.Checked)
                table.setItem(r, 0, chk_item)
            table.setItem(r, 1, QTableWidgetItem(str(k)))
            table.setItem(r, 2, QTableWidgetItem(str(v)))
        # 调整列宽：名称列自适应，值列拉伸
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        # 记录指标名 -> 行号映射，供导出时过滤
        self._vw_metric_rows = {}
        basic_n = len(basic)
        for i, k in enumerate(metrics):
            self._vw_metric_rows[k] = basic_n + i

    def refresh_viewer_plot(self, *args):
        """参数调整后的实时重绘入口（按钮/下拉框变化时触发）。"""
        if getattr(self, "vw_mono", None) is None:
            return
        self._draw_viewer_plot()

    @staticmethod
    def _get_window(name, n):
        """根据下拉框名称返回对应窗函数数组。"""
        if "矩形" in name:
            return np.ones(n)
        if "汉明" in name:
            return np.hamming(n)
        if "布莱克曼" in name:
            return np.blackman(n)
        return np.hanning(n)  # 默认汉宁

    def _draw_viewer_plot(self):
        """按当前参数在查看器画布绘图，支持窗函数/FFT点数/绘图模式/dB 实时调整。"""
        if getattr(self, "vw_mono", None) is None:
            return
        mono = self.vw_mono
        sr = self.vw_sr
        mode = self.vw_mode_cb.currentText()
        win_name = self.vw_win_cb.currentText()
        fft_sel = self.vw_fft_cb.currentText()
        use_db = self.vw_db_chk.isChecked()

        # 决定 FFT 点数：自动=取样本长度最近的 2 的幂，否则用指定值
        if fft_sel == "自动":
            nfft = 1 << int(np.log2(len(mono)))
        else:
            nfft = int(fft_sel)

        if mode == "仅波形":
            sub = self.vw_canvas.clear()
            t = np.arange(len(mono)) / sr
            sub.plot(t, mono, color="#1f77b4", width=0.8)
            sub.set_labels("波形", "时间 (s)", "幅度")
            sub.grid()
        elif mode == "仅频谱":
            sub = self.vw_canvas.clear()
            self._plot_spectrum(sub, mono, sr, nfft, win_name, use_db)
        elif mode == "语谱图":
            sub = self.vw_canvas.clear()
            self._plot_spectrogram(sub, mono, sr, nfft, win_name)
        else:  # 波形 + 频谱
            subs = self.vw_canvas.make_grid(2)
            t = np.arange(len(mono)) / sr
            subs[0].plot(t, mono, color="#1f77b4", width=0.8)
            subs[0].set_labels("波形", "时间 (s)", "幅度")
            subs[0].grid()
            self._plot_spectrum(subs[1], mono, sr, nfft, win_name, use_db)

        self.vw_canvas.finalize()

    def _plot_spectrum(self, sub, mono, sr, nfft, win_name, use_db):
        """在给定子图适配器上绘制 FFT 频谱（支持窗函数与 dB/线性幅度）。"""
        seg = mono[:nfft]
        win = self._get_window(win_name, len(seg))
        spec = np.abs(np.fft.rfft(seg * win))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        if use_db:
            spec = 20 * np.log10(np.maximum(spec, 1e-12))
            sub.plot(freqs, spec, color="#d62728", width=1)
            ylabel = "幅度 (dB)"
        else:
            sub.plot(freqs, np.maximum(spec, 1e-12), color="#d62728", width=1, style="log")
            ylabel = "幅度"
        sub.set_labels(f"频谱 (FFT {len(seg)}点, {win_name})", "频率 (Hz)", ylabel)
        sub.grid()

    def _plot_spectrogram(self, sub, mono, sr, nfft, win_name):
        """绘制语谱图 (STFT)。"""
        hop = nfft // 2
        win = self._get_window(win_name, nfft)
        n_frames = max(1, (len(mono) - nfft) // hop + 1)
        spec = []
        for i in range(n_frames):
            seg = mono[i * hop: i * hop + nfft]
            if len(seg) < nfft:
                break
            spec.append(np.abs(np.fft.rfft(seg * win)))
        if not spec:
            sub.text("音频过短，无法绘制语谱图")
            return
        S = np.array(spec).T
        S_db = 20 * np.log10(np.maximum(S, 1e-12))
        sub.imshow(S_db, extent=[0, len(mono) / sr, 0, sr / 2], cmap="magma")
        sub.set_labels(f"语谱图 ({win_name})", "时间 (s)", "频率 (Hz)")

    def _checked_metrics(self):
        """返回当前勾选（显示）的指标 dict，用于导出与显示。"""
        checked = {}
        for k, row in getattr(self, "_vw_metric_rows", {}).items():
            item = self.vw_info_table.item(row, 0)
            if item is None or item.checkState() == Qt.Checked:
                checked[k] = self.vw_metrics[k]
        return checked

    def export_viewer_result(self):
        """把当前查看器勾选的指标导出为文本/JSON 到 output 目录。"""
        if self.vw_metrics is None or not getattr(self, "vw_file", None):
            QMessageBox.warning(self, "提示", "请先打开一个音频文件")
            return
        out_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "output"))
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.vw_file))[0]
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        checked = self._checked_metrics()

        # 文本报告
        txt_path = os.path.join(out_dir, f"{base}_metrics_{ts}.txt")
        lines = [f"# 音频分析结果  {dt.datetime.now()}", f"# 文件: {self.vw_file}", ""]
        lines.append("── 基本信息 ──")
        for k, v in self.vw_basic.items():
            lines.append(f"{k}: {v}")
        lines.append("\n── 音频指标 ──")
        for k, v in checked.items():
            lines.append(f"{k}: {v}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # JSON 结果（便于后续程序化处理）
        import json
        json_path = os.path.join(out_dir, f"{base}_metrics_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"file": self.vw_file, "basic": self.vw_basic,
                       "metrics": checked}, f, ensure_ascii=False, indent=2)

        self._set_status(f"结果已导出: {txt_path}")
        # 询问是否打开导出目录
        ret = QMessageBox.question(
            self, "导出成功", f"文本: {txt_path}\nJSON: {json_path}\n\n是否打开所在目录？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            try:
                if sys.platform.startswith("win"):
                    os.startfile(out_dir)
                elif sys.platform == "darwin":
                    import subprocess; subprocess.Popen(["open", out_dir])
                else:
                    import subprocess; subprocess.Popen(["xdg-open", out_dir])
            except Exception as e:
                QMessageBox.warning(self, "提示", f"无法打开目录: {e}")


# =========================================================================
# 全局样式表 (QSS)：美化界面
# =========================================================================
APP_QSS = """
QMainWindow, QWidget { background: #f5f6fa; }
QTabWidget::pane { border: 1px solid #d0d3d9; border-radius: 4px; background: #ffffff; }
QTabBar::tab {
    background: #e8eaf0; padding: 8px 18px; border-top-left-radius: 6px;
    border-top-right-radius: 6px; margin-right: 2px; font-weight: bold; color: #4a4f57;
}
QTabBar::tab:selected { background: #2c7be5; color: #ffffff; }
QTabBar::tab:hover { background: #d5e2ff; }
QPushButton {
    background: #f0f2f7; border: 1px solid #c8ccd4; border-radius: 5px;
    padding: 6px 14px; color: #333; font-weight: 500;
}
QPushButton:hover { background: #e3e9f5; border-color: #2c7be5; }
QPushButton:pressed { background: #cdd9ee; }
QPushButton:disabled { background: #eceef2; color: #a0a5ad; border-color: #d9dce2; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    border: 1px solid #c8ccd4; border-radius: 4px; padding: 4px 8px; background: #ffffff;
}
QComboBox:hover, QLineEdit:hover { border-color: #2c7be5; }
QComboBox:focus, QLineEdit:focus { border-color: #2c7be5; }
QGroupBox {
    border: 1px solid #d0d3d9; border-radius: 6px; margin-top: 10px; background: #ffffff;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #2c7be5; font-weight: bold; }
QListWidget, QTableWidget { border: 1px solid #d0d3d9; border-radius: 4px; background: #ffffff; }
QTableWidget { gridline-color: #e4e6eb; }
QHeaderView::section {
    background: #eef1f6; padding: 4px; border: none; border-bottom: 1px solid #d0d3d9;
    font-weight: bold; color: #4a4f57;
}
QProgressBar {
    border: 1px solid #c8ccd4; border-radius: 5px; background: #ffffff; height: 14px; text-align: center;
}
QProgressBar::chunk { background: #2c7be5; border-radius: 4px; }
QSplitter::handle { background: #d0d3d9; }
"""


def _apply_style(app):
    """应用全局 QSS 样式表，美化界面。"""
    app.setStyleSheet(APP_QSS)


def main():
    if not HAS_QT:
        print("错误: PySide6 未安装，请 pip install PySide6")
        print(_qt_err)
        sys.exit(1)
    if not HAS_MPL:
        print("错误: matplotlib 未安装（GUI 绘图依赖），请 pip install matplotlib")
        sys.exit(1)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    _apply_style(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
