import sys
import numpy as np
import pywt
from scipy import signal
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGroupBox, QLabel, QComboBox,
                               QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget,
                               QScrollArea, QSizePolicy, QGridLayout)
from PySide6.QtCore import Qt
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
from PySide6.QtGui import QPainter


class WaveletEqualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("时间序列小波分析均衡器")
        self.setGeometry(100, 100, 1920, 1080)

        # 初始化变量
        self.original_signal = None
        self.sampling_rate = 1.0
        self.wavelet_coeffs = None
        self.modified_coeffs = None
        self.gain_factors = []
        self.current_wavelet = 'db4'
        self.decomposition_level = 5

        # QtCharts相关变量
        self.signal_charts = []
        self.signal_chart_views = []
        self.coeff_charts = []
        self.coeff_chart_views = []
        self.spectrum_charts = []
        self.spectrum_chart_views = []

        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = self.create_control_panel()

        # 右侧显示区域
        display_panel = self.create_display_panel()

        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(display_panel)
        splitter.setSizes([300, 1100])

        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_control_panel(self):
        """创建左侧控制面板"""
        control_widget = QWidget()
        layout = QVBoxLayout()

        # 数据加载组
        data_group = QGroupBox("数据加载")
        data_layout = QVBoxLayout()

        self.load_btn = QPushButton("加载时间序列数据")
        self.load_btn.clicked.connect(self.load_data)
        data_layout.addWidget(self.load_btn)

        # 示例数据生成
        self.generate_btn = QPushButton("生成示例数据")
        self.generate_btn.clicked.connect(self.generate_sample_data)
        data_layout.addWidget(self.generate_btn)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # 小波参数组
        wavelet_group = QGroupBox("小波分析参数")
        wavelet_layout = QVBoxLayout()

        # 小波基选择
        wavelet_layout.addWidget(QLabel("选择小波基:"))
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(['db4', 'db6', 'sym4', 'sym6'])
        self.wavelet_combo.currentTextChanged.connect(self.wavelet_changed)
        wavelet_layout.addWidget(self.wavelet_combo)

        # 分解层数
        wavelet_layout.addWidget(QLabel("分解层数:"))
        self.level_spin = QSpinBox()
        self.level_spin.setRange(5, 12)
        self.level_spin.setValue(5)
        self.level_spin.valueChanged.connect(self.decomposition_level_changed)
        wavelet_layout.addWidget(self.level_spin)

        # 执行分解按钮
        self.decompose_btn = QPushButton("执行小波分解")
        self.decompose_btn.clicked.connect(self.perform_decomposition)
        self.decompose_btn.setEnabled(False)
        wavelet_layout.addWidget(self.decompose_btn)

        wavelet_group.setLayout(wavelet_layout)
        layout.addWidget(wavelet_group)

        # 增益控制组
        self.gain_group = QGroupBox("小波系数增益控制")
        self.gain_layout = QVBoxLayout()
        self.gain_group.setLayout(self.gain_layout)
        self.gain_group.setEnabled(False)
        layout.addWidget(self.gain_group)

        # 重置和重构按钮
        self.reset_btn = QPushButton("重置增益")
        self.reset_btn.clicked.connect(self.reset_gains)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn)

        self.reconstruct_btn = QPushButton("重构信号")
        self.reconstruct_btn.clicked.connect(self.reconstruct_signal)
        self.reconstruct_btn.setEnabled(False)
        layout.addWidget(self.reconstruct_btn)

        layout.addStretch()
        control_widget.setLayout(layout)
        return control_widget

    def create_display_panel(self):
        """创建右侧显示面板"""
        display_widget = QWidget()
        layout = QVBoxLayout()

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 信号显示标签页
        self.signal_tab = QWidget()
        self.signal_layout = QVBoxLayout()

        # 创建信号图表容器（用于多个子图）
        self.signal_charts_container = QWidget()
        self.signal_charts_layout = QVBoxLayout(self.signal_charts_container)

        # 添加滚动区域
        signal_scroll = QScrollArea()
        signal_scroll.setWidget(self.signal_charts_container)
        signal_scroll.setWidgetResizable(True)
        self.signal_layout.addWidget(signal_scroll)

        self.signal_tab.setLayout(self.signal_layout)
        self.tab_widget.addTab(self.signal_tab, "信号分析")

        # 小波系数标签页
        self.coeff_tab = QWidget()
        self.coeff_layout = QVBoxLayout()

        # 创建小波系数图表容器
        self.coeff_charts_container = QWidget()
        self.coeff_charts_layout = QVBoxLayout(self.coeff_charts_container)

        # 添加滚动区域
        coeff_scroll = QScrollArea()
        coeff_scroll.setWidget(self.coeff_charts_container)
        coeff_scroll.setWidgetResizable(True)
        self.coeff_layout.addWidget(coeff_scroll)

        self.coeff_tab.setLayout(self.coeff_layout)
        self.tab_widget.addTab(self.coeff_tab, "小波系数")

        # 频谱分析标签页
        self.spectrum_tab = QWidget()
        self.spectrum_layout = QVBoxLayout()

        # 创建频谱图表容器
        self.spectrum_charts_container = QWidget()
        self.spectrum_charts_layout = QVBoxLayout(self.spectrum_charts_container)

        # 添加滚动区域
        spectrum_scroll = QScrollArea()
        spectrum_scroll.setWidget(self.spectrum_charts_container)
        spectrum_scroll.setWidgetResizable(True)
        self.spectrum_layout.addWidget(spectrum_scroll)

        self.spectrum_tab.setLayout(self.spectrum_layout)
        self.tab_widget.addTab(self.spectrum_tab, "频谱分析")

        layout.addWidget(self.tab_widget)
        display_widget.setLayout(layout)
        return display_widget

    def create_gain_controls(self):
        """创建增益控制滑块"""
        # 清除现有控件
        for i in reversed(range(self.gain_layout.count())):
            widget = self.gain_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.gain_sliders = []
        self.gain_labels = []

        if self.wavelet_coeffs is None:
            return

        n_levels = len(self.wavelet_coeffs)

        for i in range(n_levels):
            level_label = f"Level {i}" if i < n_levels - 1 else "Approx"
            h_layout = QHBoxLayout()

            label = QLabel(f"{level_label}: ")
            h_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 200)  # 0% 到 200%
            slider.setValue(100)  # 默认100%
            slider.valueChanged.connect(self.gain_changed)
            h_layout.addWidget(slider)

            value_label = QLabel("100%")
            h_layout.addWidget(value_label)

            self.gain_sliders.append(slider)
            self.gain_labels.append(value_label)

            container = QWidget()
            container.setLayout(h_layout)
            self.gain_layout.addWidget(container)

        self.gain_factors = [1.0] * n_levels

    def load_data(self):
        """加载时间序列数据"""
        # 这里简化实现，实际应用中应该添加文件对话框
        # 生成示例数据代替文件加载
        self.generate_sample_data()

    def generate_sample_data(self):
        """生成示例时间序列数据"""
        import mat_reader
        data = mat_reader.MatData("/home/transen5/Documents/MTData/031BE-20240501-20240520-dt5_struct.mat")
        self.original_signal = data.field_data['Bx'][2000:5000].interpolate(method='polynomial', order=2)
        self.sampling_rate = 1 / data.dt
        self.decompose_btn.setEnabled(True)
        self.plot_original_signal()

    def wavelet_changed(self, wavelet_name):
        """小波基改变时的处理"""
        self.current_wavelet = wavelet_name
        if self.original_signal is not None:
            self.perform_decomposition()

    def decomposition_level_changed(self, level):
        """分解层数改变时的处理"""
        self.decomposition_level = level
        if self.original_signal is not None:
            self.perform_decomposition()

    def perform_decomposition(self):
        """执行小波分解"""
        try:
            # 计算最大可用分解层数
            max_level = pywt.dwt_max_level(len(self.original_signal),
                                           pywt.Wavelet(self.current_wavelet).dec_len)
            level = min(self.decomposition_level, max_level)

            # 执行小波分解
            self.wavelet_coeffs = pywt.wavedec(self.original_signal,
                                               self.current_wavelet,
                                               level=level)

            self.modified_coeffs = [coeff.copy() for coeff in self.wavelet_coeffs]

            self.create_gain_controls()
            self.gain_group.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.reconstruct_btn.setEnabled(True)

            self.plot_wavelet_coefficients()
            self.plot_spectral_analysis()

        except Exception as e:
            print(f"小波分解错误: {e}")

    def gain_changed(self):
        """增益改变时的处理"""
        for i, slider in enumerate(self.gain_sliders):
            gain_factor = slider.value() / 100.0
            self.gain_factors[i] = gain_factor
            self.gain_labels[i].setText(f"{slider.value()}%")

            # 应用增益到系数
            if i < len(self.modified_coeffs):
                self.modified_coeffs[i] = self.wavelet_coeffs[i] * gain_factor

    def reset_gains(self):
        """重置所有增益到100%"""
        for slider in self.gain_sliders:
            slider.setValue(100)
        self.gain_changed()

    def reconstruct_signal(self):
        """重构信号"""
        try:
            # 使用修改后的小波系数重构信号
            reconstructed = pywt.waverec(self.modified_coeffs, self.current_wavelet)

            # 确保重构信号长度与原始信号一致
            min_len = min(len(reconstructed), len(self.original_signal))
            reconstructed = reconstructed[:min_len]
            original = self.original_signal[:min_len]

            self.plot_signal_comparison(original, reconstructed)

        except Exception as e:
            print(f"信号重构错误: {e}")

    def clear_charts_container(self, container_layout):
        """清除图表容器中的所有图表"""
        # 清除所有图表视图
        for i in reversed(range(container_layout.count())):
            item = container_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

    def plot_original_signal(self):
        """绘制原始信号"""
        if self.original_signal is None:
            return

        # 清除现有图表
        self.clear_charts_container(self.signal_charts_layout)
        self.signal_charts = []
        self.signal_chart_views = []

        # 创建图表
        chart = QChart()
        chart.setTitle("原始时间序列信号")
        chart.legend().setVisible(True)

        # 创建数据系列
        series = QLineSeries()
        series.setName("原始信号")

        # 添加数据点
        t = np.arange(len(self.original_signal)) / self.sampling_rate
        for i in range(len(self.original_signal)):
            series.append(t[i], self.original_signal.iloc[i] if hasattr(self.original_signal, 'iloc') else
            self.original_signal[i])

        # 添加到图表
        chart.addSeries(series)

        # 创建坐标轴
        axis_x = QValueAxis()
        axis_x.setTitleText("时间 (秒)")
        axis_x.setLabelFormat("%.1f")

        axis_y = QValueAxis()
        axis_y.setTitleText("幅值")
        axis_y.setLabelFormat("%.3f")

        # 设置坐标轴范围
        axis_x.setRange(0, t[-1])
        y_min = self.original_signal.min()
        y_max = self.original_signal.max()
        y_range = y_max - y_min
        axis_y.setRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # 添加坐标轴到图表
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)

        # 关联系列到坐标轴
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)

        # 创建图表视图
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setMinimumHeight(400)

        # 添加到容器
        self.signal_charts_layout.addWidget(chart_view)

        # 保存引用
        self.signal_charts.append(chart)
        self.signal_chart_views.append(chart_view)

    def plot_wavelet_coefficients(self):
        """绘制小波系数"""
        if self.wavelet_coeffs is None:
            return

        # 清除现有图表
        self.clear_charts_container(self.coeff_charts_layout)
        self.coeff_charts = []
        self.coeff_chart_views = []

        n_levels = len(self.wavelet_coeffs)

        # 为每个级别创建图表
        for i, coeff in enumerate(self.wavelet_coeffs):
            chart = QChart()
            level_type = "近似系数" if i == n_levels - 1 else f"细节系数 Level {n_levels - 1 - i}"
            chart.setTitle(level_type)
            chart.legend().setVisible(False)

            # 创建数据系列
            series = QLineSeries()

            # 添加数据点
            for j in range(len(coeff)):
                series.append(j, coeff[j])

            # 添加到图表
            chart.addSeries(series)

            # 创建坐标轴
            axis_x = QValueAxis()
            axis_x.setTitleText("系数索引")

            axis_y = QValueAxis()
            axis_y.setTitleText("系数值")
            axis_y.setLabelFormat("%.3f")

            # 添加到图表
            chart.addAxis(axis_x, Qt.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignLeft)

            # 关联系列到坐标轴
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

            # 创建图表视图
            chart_view = QChartView(chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            chart_view.setMinimumHeight(300)

            # 添加到容器
            self.coeff_charts_layout.addWidget(chart_view)

            # 保存引用
            self.coeff_charts.append(chart)
            self.coeff_chart_views.append(chart_view)

    def plot_spectral_analysis(self):
        """绘制频谱分析"""
        if self.original_signal is None:
            return

        # 清除现有图表
        self.clear_charts_container(self.spectrum_charts_layout)
        self.spectrum_charts = []
        self.spectrum_chart_views = []

        # 创建第一个子图：功率谱密度
        chart1 = QChart()
        chart1.setTitle("原始信号功率谱密度")
        chart1.legend().setVisible(True)

        # 计算原始信号的频谱
        f, Pxx = signal.welch(self.original_signal, self.sampling_rate, nperseg=256)

        # 创建功率谱密度系列
        psd_series = QLineSeries()
        psd_series.setName("功率谱密度")

        # 添加数据点
        for i in range(len(f)):
            psd_series.append(f[i], Pxx[i])

        # 添加到图表
        chart1.addSeries(psd_series)

        # 创建坐标轴
        axis_x1 = QValueAxis()
        axis_x1.setTitleText("频率 [Hz]")
        axis_x1.setLabelFormat("%.2f")

        axis_y1 = QValueAxis()
        axis_y1.setTitleText("PSD [V**2/Hz]")
        axis_y1.setLabelFormat("%.2e")

        # 设置坐标轴范围
        axis_x1.setRange(0, f[-1])
        axis_y1.setRange(Pxx.min(), Pxx.max())

        # 添加到图表
        chart1.addAxis(axis_x1, Qt.AlignBottom)
        chart1.addAxis(axis_y1, Qt.AlignLeft)

        # 关联系列到坐标轴
        psd_series.attachAxis(axis_x1)
        psd_series.attachAxis(axis_y1)

        # 创建图表视图
        chart_view1 = QChartView(chart1)
        chart_view1.setRenderHint(QPainter.Antialiasing)
        chart_view1.setMinimumHeight(400)

        # 添加到容器
        self.spectrum_charts_layout.addWidget(chart_view1)

        # 保存引用
        self.spectrum_charts.append(chart1)
        self.spectrum_chart_views.append(chart_view1)

        # 创建第二个子图：小波尺度图（简化版）
        chart2 = QChart()
        chart2.setTitle("小波变换能量分布")
        chart2.legend().setVisible(True)

        # 计算小波变换能量分布（简化版）
        # 这里我们使用离散小波变换的能量分布作为替代
        energy_series = QLineSeries()
        energy_series.setName("各层能量")

        # 计算各层小波系数的能量
        if self.wavelet_coeffs is not None:
            energies = []
            levels = []
            for i, coeff in enumerate(self.wavelet_coeffs):
                energies.append(np.sum(coeff ** 2))
                levels.append(i)

            # 添加数据点
            for i, energy in enumerate(energies):
                energy_series.append(i, energy)

            # 添加到图表
            chart2.addSeries(energy_series)

            # 创建坐标轴
            axis_x2 = QValueAxis()
            axis_x2.setTitleText("分解层数")
            axis_x2.setLabelFormat("%d")

            axis_y2 = QValueAxis()
            axis_y2.setTitleText("能量")
            axis_y2.setLabelFormat("%.2e")

            # 设置坐标轴范围
            axis_x2.setRange(0, len(energies) - 1)
            axis_y2.setRange(min(energies), max(energies))

            # 添加到图表
            chart2.addAxis(axis_x2, Qt.AlignBottom)
            chart2.addAxis(axis_y2, Qt.AlignLeft)

            # 关联系列到坐标轴
            energy_series.attachAxis(axis_x2)
            energy_series.attachAxis(axis_y2)

            # 创建图表视图
            chart_view2 = QChartView(chart2)
            chart_view2.setRenderHint(QPainter.Antialiasing)
            chart_view2.setMinimumHeight(400)

            # 添加到容器
            self.spectrum_charts_layout.addWidget(chart_view2)

            # 保存引用
            self.spectrum_charts.append(chart2)
            self.spectrum_chart_views.append(chart_view2)

    def plot_signal_comparison(self, original, reconstructed):
        """绘制原始信号与重构信号的对比"""
        if original is None or reconstructed is None:
            return

        # 清除现有图表
        self.clear_charts_container(self.signal_charts_layout)
        self.signal_charts = []
        self.signal_chart_views = []

        # 创建第一个子图：信号对比
        chart1 = QChart()
        chart1.setTitle("信号对比")
        chart1.legend().setVisible(True)

        # 创建原始信号系列
        original_series = QLineSeries()
        original_series.setName("原始信号")

        # 创建重构信号系列
        reconstructed_series = QLineSeries()
        reconstructed_series.setName("重构信号")

        # 添加数据点
        t = np.arange(len(original)) / self.sampling_rate
        for i in range(len(original)):
            original_series.append(t[i], original[i])
            reconstructed_series.append(t[i], reconstructed[i])

        # 设置不同颜色
        original_series.setColor(Qt.blue)
        reconstructed_series.setColor(Qt.red)

        # 添加到图表
        chart1.addSeries(original_series)
        chart1.addSeries(reconstructed_series)

        # 创建坐标轴
        axis_x1 = QValueAxis()
        axis_x1.setTitleText("时间 (秒)")
        axis_x1.setLabelFormat("%.1f")

        axis_y1 = QValueAxis()
        axis_y1.setTitleText("幅值")
        axis_y1.setLabelFormat("%.3f")

        # 设置坐标轴范围
        axis_x1.setRange(0, t[-1])
        y_min = min(original.min(), reconstructed.min())
        y_max = max(original.max(), reconstructed.max())
        y_range = y_max - y_min
        axis_y1.setRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # 添加到图表
        chart1.addAxis(axis_x1, Qt.AlignBottom)
        chart1.addAxis(axis_y1, Qt.AlignLeft)

        # 关联系列到坐标轴
        original_series.attachAxis(axis_x1)
        original_series.attachAxis(axis_y1)
        reconstructed_series.attachAxis(axis_x1)
        reconstructed_series.attachAxis(axis_y1)

        # 创建图表视图
        chart_view1 = QChartView(chart1)
        chart_view1.setRenderHint(QPainter.Antialiasing)
        chart_view1.setMinimumHeight(400)

        # 添加到容器
        self.signal_charts_layout.addWidget(chart_view1)

        # 保存引用
        self.signal_charts.append(chart1)
        self.signal_chart_views.append(chart_view1)

        # 创建第二个子图：残差
        chart2 = QChart()
        chart2.setTitle("残差")
        chart2.legend().setVisible(True)

        # 创建残差系列
        residual_series = QLineSeries()
        residual_series.setName("残差")

        # 计算残差
        residual = original - reconstructed
        rms = np.sqrt(np.mean(residual ** 2))

        # 添加数据点
        for i in range(len(residual)):
            residual_series.append(t[i], residual[i])

        # 设置颜色
        residual_series.setColor(Qt.green)

        # 添加到图表
        chart2.addSeries(residual_series)

        # 创建坐标轴
        axis_x2 = QValueAxis()
        axis_x2.setTitleText("时间 (秒)")
        axis_x2.setLabelFormat("%.1f")

        axis_y2 = QValueAxis()
        axis_y2.setTitleText("幅值")
        axis_y2.setLabelFormat("%.3f")

        # 设置坐标轴范围
        axis_x2.setRange(0, t[-1])
        residual_min = residual.min()
        residual_max = residual.max()
        residual_range = residual_max - residual_min
        axis_y2.setRange(residual_min - 0.1 * residual_range, residual_max + 0.1 * residual_range)

        # 添加到图表
        chart2.addAxis(axis_x2, Qt.AlignBottom)
        chart2.addAxis(axis_y2, Qt.AlignLeft)

        # 关联系列到坐标轴
        residual_series.attachAxis(axis_x2)
        residual_series.attachAxis(axis_y2)

        # 创建图表视图
        chart_view2 = QChartView(chart2)
        chart_view2.setRenderHint(QPainter.Antialiasing)
        chart_view2.setMinimumHeight(400)

        # 添加到容器
        self.signal_charts_layout.addWidget(chart_view2)

        # 保存引用
        self.signal_charts.append(chart2)
        self.signal_chart_views.append(chart_view2)

        # 更新图表标题
        chart2.setTitle(f"残差 (RMS: {rms:.4f})")


def main():
    app = QApplication(sys.argv)
    window = WaveletEqualizerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()