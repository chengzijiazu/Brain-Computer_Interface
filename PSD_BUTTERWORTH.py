import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, FilterTypes
import serial
import matplotlib.pyplot as plt

# 设置 Cyton 开发板参数
params = BrainFlowInputParams()
params.serial_port = "COM3"  # 根据你的设备修改串口号

# 初始化 Cyton 开发板
board_id = BoardIds.CYTON_BOARD
board = BoardShim(board_id, params)

# 启动会话
board.prepare_session()
board.start_stream()

# 定义各波段的频段
DELTA_BAND = [1, 4]
THETA_BAND = [4, 8]
ALPHA_BAND = [8, 12]
BETA_BAND = [12, 30]
GAMMA_BAND = [30, 40]  # 调整 Gamma 波段范围

# 定义采样率和通道数
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)  # 获取 8 个 EEG 通道

# 初始化串口通信（用于控制 LED）
arduino = serial.Serial('COM17', 9600, timeout=3)  # 根据 Arduino 的串口号修改
time.sleep(2)  # 等待串口初始化

def send_led_command(state):
    """向 Arduino 发送 LED 控制指令"""
    arduino.write(str(state).encode())

def preprocess_data(data, scale_factor=1.0):
    """预处理 EEG 数据：缩放、去偏移、带通滤波、带阻滤波"""
    # 缩放数据
    data = data * scale_factor

    # 去除直流偏移
    data = data - np.mean(data, axis=1, keepdims=True)

    # 带通滤波 (1-50 Hz)
    for channel in range(data.shape[0]):
        DataFilter.perform_bandpass(data[channel], sampling_rate, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)

    # 带阻滤波 (49-51 Hz，去除电源线干扰)
    for channel in range(data.shape[0]):
        DataFilter.perform_bandstop(data[channel], sampling_rate, 49.0, 51.0, 4, FilterTypes.BUTTERWORTH.value, 0)

    return data

def calculate_band_powers(data):
    """计算各波段的功率"""
    delta_powers = []
    theta_powers = []
    alpha_powers = []
    beta_powers = []
    gamma_powers = []

    for channel in range(data.shape[0]):
        # 计算 PSD
        psd = DataFilter.get_psd_welch(data[channel], nfft=256, overlap=128, sampling_rate=sampling_rate, window=WindowOperations.HANNING.value)

        # 提取各波段功率
        delta_power = DataFilter.get_band_power(psd, DELTA_BAND[0], DELTA_BAND[1])
        theta_power = DataFilter.get_band_power(psd, THETA_BAND[0], THETA_BAND[1])
        alpha_power = DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1])
        beta_power = DataFilter.get_band_power(psd, BETA_BAND[0], BETA_BAND[1])
        gamma_power = DataFilter.get_band_power(psd, GAMMA_BAND[0], GAMMA_BAND[1])

        # 将各波段功率添加到列表中
        delta_powers.append(delta_power)
        theta_powers.append(theta_power)
        alpha_powers.append(alpha_power)
        beta_powers.append(beta_power)
        gamma_powers.append(gamma_power)

    # 计算平均功率
    avg_delta_power = np.mean(delta_powers)
    avg_theta_power = np.mean(theta_powers)
    avg_alpha_power = np.mean(alpha_powers)
    avg_beta_power = np.mean(beta_powers)
    avg_gamma_power = np.mean(gamma_powers)

    return avg_delta_power, avg_theta_power, avg_alpha_power, avg_beta_power, avg_gamma_power

try:
    while True:
        # 获取最新的 256 个样本
        data = board.get_current_board_data(256)

        # 提取 EEG 数据
        eeg_data = data[eeg_channels, :]

        # 预处理 EEG 数据
        eeg_data = preprocess_data(eeg_data, scale_factor=1.0)  # 根据开发板文档设置正确的缩放因子

        # 绘制原始信号和滤波后信号（调试用）
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(data[eeg_channels[0], :])
        plt.title("Raw EEG Signal (Channel 1)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude (µV)")

        plt.subplot(2, 1, 2)
        plt.plot(eeg_data[0, :])
        plt.title("Filtered EEG Signal (Channel 1)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude (µV)")

        plt.tight_layout()
        plt.show()

        # 计算各波段的平均功率
        avg_delta_power, avg_theta_power, avg_alpha_power, avg_beta_power, avg_gamma_power = calculate_band_powers(eeg_data)

        # 打印各波段的平均功率
        print(f"Average Delta Power: {avg_delta_power:.2f}")
        print(f"Average Theta Power: {avg_theta_power:.2f}")
        print(f"Average Alpha Power: {avg_alpha_power:.2f}")
        print(f"Average Beta Power: {avg_beta_power:.2f}")
        print(f"Average Gamma Power: {avg_gamma_power:.2f}")
        print("----------------------------------------")

        # 控制 LED
        if avg_alpha_power > 10:  # 阈值可根据实际情况调整
            send_led_command(1)  # 发送亮灯指令
        else:
            send_led_command(0)  # 发送灭灯指令

        # 等待一段时间
        time.sleep(1)

except KeyboardInterrupt:
    print("Streaming stopped.")

finally:
    # 停止会话
    board.stop_stream()
    board.release_session()
    arduino.close()  # 关闭串口