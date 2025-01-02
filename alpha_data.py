import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations
import serial

# 设置 Cyton 开发板参数
params = BrainFlowInputParams()
params.serial_port = "COM3"  # 根据你的设备修改串口号

# 初始化 Cyton 开发板
board_id = BoardIds.CYTON_BOARD
board = BoardShim(board_id, params)

# 启动会话
board.prepare_session()
board.start_stream()

# 定义 Alpha 波频段
ALPHA_BAND = [8, 12]

# 定义采样率和通道数
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)  # 获取 8 个 EEG 通道

# 初始化串口通信（用于控制 LED）
arduino = serial.Serial('COM17', 9600, timeout=3)  # 根据 Arduino 的串口号修改
time.sleep(2)  # 等待串口初始化

def send_led_command(state):
    """向 Arduino 发送 LED 控制指令"""
    arduino.write(str(state).encode())

try:
    while True:
        # 获取最新的 256 个样本
        data = board.get_current_board_data(256)

        # 提取 EEG 数据
        eeg_data = data[eeg_channels, :]

        # 计算每个通道的 Alpha 波功率
        alpha_powers = []
        for channel in eeg_channels:
            # 计算 PSD
            psd = DataFilter.get_psd_welch(data[channel], nfft=256, overlap=128, sampling_rate=sampling_rate, window=WindowOperations.HANNING.value)
            # 提取 Alpha 波功率
            alpha_power = DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1])
            alpha_powers.append(alpha_power)

        # 计算平均 Alpha 波功率
        avg_alpha_power = np.mean(alpha_powers)
        print(f"Average Alpha Power: {avg_alpha_power:.2f}")

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