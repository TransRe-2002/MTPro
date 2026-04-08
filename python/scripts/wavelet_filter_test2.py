import sys
import numpy as np
import pywt
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGroupBox, QLabel, QComboBox,
                               QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget)
from PySide6.QtCore import Qt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
        self.signal_fig = Figure(figsize=(10, 8))
        self.signal_canvas = FigureCanvas(self.signal_fig)
        self.signal_layout.addWidget(self.signal_canvas)
        self.signal_tab.setLayout(self.signal_layout)
        self.tab_widget.addTab(self.signal_tab, "信号分析")

        # 小波系数标签页
        self.coeff_tab = QWidget()
        self.coeff_layout = QVBoxLayout()
        self.coeff_fig = Figure(figsize=(10, 8))
        self.coeff_canvas = FigureCanvas(self.coeff_fig)
        self.coeff_layout.addWidget(self.coeff_canvas)
        self.coeff_tab.setLayout(self.coeff_layout)
        self.tab_widget.addTab(self.coeff_tab, "小波系数")

        # 频谱分析标签页
        self.spectrum_tab = QWidget()
        self.spectrum_layout = QVBoxLayout()
        self.spectrum_fig = Figure(figsize=(10, 8))
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_layout.addWidget(self.spectrum_canvas)
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
        self.original_signal = data.field_data['Bx'][0:-1].interpolate(method='polynomial', order=2)
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
            # 计算最大可用分解层数[3](@ref)
            max_level = pywt.dwt_max_level(len(self.original_signal),
                                           pywt.Wavelet(self.current_wavelet).dec_len)
            level = min(self.decomposition_level, max_level)

            # 执行小波分解[2,3](@ref)
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

            # 应用增益到系数[4](@ref)
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
            # 使用修改后的小波系数重构信号[3](@ref)
            reconstructed = pywt.waverec(self.modified_coeffs, self.current_wavelet)

            # 确保重构信号长度与原始信号一致
            min_len = min(len(reconstructed), len(self.original_signal))
            reconstructed = reconstructed[:min_len]
            original = self.original_signal[:min_len]

            self.plot_signal_comparison(original, reconstructed)

        except Exception as e:
            print(f"信号重构错误: {e}")

    def plot_original_signal(self):
        """绘制原始信号"""
        self.signal_fig.clear()
        ax = self.signal_fig.add_subplot(111)
        t = np.arange(len(self.original_signal)) / self.sampling_rate

        ax.plot(t, self.original_signal, 'b-', linewidth=1, label='原始信号')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('幅值')
        ax.set_title('原始时间序列信号')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.signal_canvas.draw()

    def plot_wavelet_coefficients(self):
        """绘制小波系数"""
        self.coeff_fig.clear()

        n_levels = len(self.wavelet_coeffs)
        axes = self.coeff_fig.subplots(n_levels, 1)

        if n_levels == 1:
            axes = [axes]

        for i, (coeff, ax) in enumerate(zip(self.wavelet_coeffs, axes)):
            level_type = "近似系数" if i == n_levels - 1 else f"细节系数 Level {i}"
            ax.plot(coeff, 'b-', linewidth=0.8)
            ax.set_xticks([])
            ax.set_ylabel(level_type)
            ax.grid(True, alpha=0.3)

            if i == n_levels - 1:
                ax.set_xlabel('系数索引')

        self.coeff_fig.suptitle('小波分解系数', fontsize=12)
        self.coeff_fig.tight_layout()
        self.coeff_canvas.draw()

    def plot_spectral_analysis(self):
        """绘制频谱分析"""
        self.spectrum_fig.clear()
        axes = self.spectrum_fig.subplots(2, 1)

        # 计算原始信号的频谱[7](@ref)
        f, Pxx = signal.welch(self.original_signal, self.sampling_rate,
                              nperseg=256)

        axes[0].semilogy(f, Pxx, 'b-', linewidth=1)
        axes[0].set_xlabel('频率 [Hz]')
        axes[0].set_ylabel('PSD [V**2/Hz]')
        axes[0].set_title('原始信号功率谱密度')
        axes[0].grid(True, alpha=0.3)

        # 绘制小波尺度图
        if len(self.original_signal) > 0:
            scales = np.arange(1, 65)
            coefficients, frequencies = pywt.cwt(self.original_signal,
                                                 scales, self.current_wavelet)

            im = axes[1].imshow(np.abs(coefficients), aspect='auto',
                                extent=[0, len(self.original_signal) / self.sampling_rate,
                                        1, 64],
                                cmap='viridis')
            axes[1].set_ylabel('尺度')
            axes[1].set_xlabel('时间 [秒]')
            axes[1].set_title('连续小波变换尺度图')
            self.spectrum_fig.colorbar(im, ax=axes[1])

        self.spectrum_fig.tight_layout()
        self.spectrum_canvas.draw()

    def plot_signal_comparison(self, original, reconstructed):
        """绘制原始信号与重构信号的对比"""
        self.signal_fig.clear()
        axes = self.signal_fig.subplots(2, 1)

        t = np.arange(len(original)) / self.sampling_rate

        # 绘制信号对比
        axes[0].plot(t, original, 'b-', linewidth=1, label='原始信号', alpha=0.7)
        axes[0].plot(t, reconstructed, 'r-', linewidth=1, label='重构信号', alpha=0.7)
        axes[0].set_xlabel('时间 (秒)')
        axes[0].set_ylabel('幅值')
        axes[0].set_title('信号对比')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 绘制残差
        residual = original - reconstructed
        axes[1].plot(t, residual, 'g-', linewidth=1, label='残差')
        axes[1].set_xlabel('时间 (秒)')
        axes[1].set_ylabel('幅值')
        axes[1].set_title(f'残差 (RMS: {np.sqrt(np.mean(residual ** 2)):.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        self.signal_fig.tight_layout()
        self.signal_canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    window = WaveletEqualizerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()