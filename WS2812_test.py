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

def send_led_command(r, g, b):
    """向 Arduino 发送 RGB 控制指令"""
    command = f"{r},{g},{b}\n"
    arduino.write(command.encode())

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

def normalize_to_rgb(value, min_val=0, max_val=1):
    """将功率值归一化到 0-255 范围"""
    return int(np.clip(255 * (value - min_val) / (max_val - min_val), 0, 255))

def calculate_rgb(avg_delta, avg_theta, avg_alpha, avg_beta, avg_gamma):
    """根据波段功率计算 RGB 值"""
    # 设置波段的影响权重，调整这些值来优化颜色映射
    weight_delta = 0.2
    weight_theta = 0.1
    weight_alpha = 0.3
    weight_beta = 0.25
    weight_gamma = 0.15

    # 计算加权的 RGB 分量
    r = normalize_to_rgb(weight_gamma * avg_gamma + weight_beta * avg_beta, 0, 100)
    g = normalize_to_rgb(weight_alpha * avg_alpha + weight_theta * avg_theta, 0, 50)
    b = normalize_to_rgb(weight_delta * avg_delta, 0, 150)

    # 确保 RGB 分量在合理范围内
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    return r, g, b

try:
    while True:
        # 获取最新的 256 个样本
        data = board.get_current_board_data(256)

        # 提取 EEG 数据
        eeg_data = data[eeg_channels, :]

        # 预处理 EEG 数据
        eeg_data = preprocess_data(eeg_data, scale_factor=1.0)  # 根据开发板文档设置正确的缩放因子

        # 计算各波段的平均功率
        avg_delta_power, avg_theta_power, avg_alpha_power, avg_beta_power, avg_gamma_power = calculate_band_powers(eeg_data)

        # 打印各波段的平均功率
        print(f"Average Delta Power: {avg_delta_power:.2f}")
        print(f"Average Theta Power: {avg_theta_power:.2f}")
        print(f"Average Alpha Power: {avg_alpha_power:.2f}")
        print(f"Average Beta Power: {avg_beta_power:.2f}")
        print(f"Average Gamma Power: {avg_gamma_power:.2f}")
        print("----------------------------------------")

        # 计算 RGB 值
        r, g, b = calculate_rgb(avg_delta_power, avg_theta_power, avg_alpha_power, avg_beta_power, avg_gamma_power)

        # 打印 RGB 值
        print(f"Calculated RGB values: R={r}, G={g}, B={b}")

        # 控制 LED
        send_led_command(r, g, b)

        # 等待一段时间
        time.sleep(1)

except KeyboardInterrupt:
    print("Streaming stopped.")

finally:
    # 停止会话
    board.stop_stream()
    board.release_session()
    arduino.close()  # 关闭串口
