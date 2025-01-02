import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations
import serial
import sys
import logging
from datetime import datetime

# 配置日志，仅记录 ERROR 级别及以上的信息到 error.log 文件
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# 设置 Cyton 开发板参数
params = BrainFlowInputParams()
params.serial_port = "COM3"  # 根据您的设备修改串口号

# 初始化 Cyton 开发板
board_id = BoardIds.CYTON_BOARD
board = BoardShim(board_id, params)

# 初始化与 Arduino 的串口通信
try:
    arduino = serial.Serial('COM17', 9600, timeout=5)  # 确保串口号正确
    time.sleep(2)  # 等待串口初始化
    # logging.info("Arduino 连接成功。")  # 移除或注释掉
except serial.SerialException as e:
    logging.error(f"连接 Arduino 失败，端口 COM17: {e}")
    sys.exit(1)  # 如果串口连接失败，退出脚本

def send_led_command(state):
    """向 Arduino 发送 LED 控制指令。"""
    try:
        arduino.write(str(state).encode())
        # logging.debug(f"发送 LED 状态: {state}")  # 移除或注释掉
    except serial.SerialException as e:
        logging.error(f"发送 LED 指令失败: {e}")

def remove_artifacts_with_wavelet(data):
    """使用小波变换去除眼动伪迹。"""
    try:
        # 确保传入的数据是 NumPy 数组
        if not isinstance(data, np.ndarray):
            logging.error(f"传入的数据不是 NumPy 数组，类型为: {type(data)}")
            return data  # 返回原始数据

        # 正确的参数顺序：数据, 小波名称, 分解层数
        wavelet_coeffs = DataFilter.perform_wavelet_transform(data, 'db4', 3)

        # 零化前两个细节系数
        for i in range(2):
            wavelet_coeffs[i] = np.zeros_like(wavelet_coeffs[i])

        # 正确的参数顺序：系数, 小波名称, 分解层数
        denoised_data = DataFilter.perform_inverse_wavelet_transform(wavelet_coeffs, 'db4', 3)
        return denoised_data
    except Exception as e:
        logging.error(f"Wavelet artifact removal failed: {e}")
        return data  # 处理失败时返回原始数据

try:
    # 准备并启动 BrainFlow 会话
    board.prepare_session()
    board.start_stream()
    # logging.info("BrainFlow 会话已启动。")  # 移除或注释掉

    # 定义 Alpha 波频段
    ALPHA_BAND = [8, 12]

    # 获取采样率和 EEG 通道
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    # logging.info(f"采样率: {sampling_rate} Hz")  # 移除或注释掉
    # logging.info(f"EEG 通道: {eeg_channels}")  # 移除或注释掉

    while True:
        # 获取最新的 256 个样本
        data = board.get_current_board_data(256)

        # 记录接收到的通道数量
        total_channels = data.shape[0]
        expected_channels = len(eeg_channels)
        # logging.debug(f"接收到的数据包含 {total_channels} 个通道。")  # 移除或注释掉

        # 验证数据形状
        if total_channels < expected_channels:
            # logging.warning(f"接收到的数据通道数 {total_channels} 少于预期的 {expected_channels}。")  # 移除或注释掉
            continue  # 跳过本次处理

        # 可选：记录 EEG 通道索引
        # logging.debug(f"EEG 通道索引: {eeg_channels}")  # 移除或注释掉

        # 提取 EEG 数据
        eeg_data = data[eeg_channels, :]

        # 确保 eeg_data 是二维 NumPy 数组
        if not isinstance(eeg_data, np.ndarray) or eeg_data.ndim != 2:
            # logging.error(f"提取的 EEG 数据不是二维 NumPy 数组，类型为: {type(eeg_data)}")  # 移除或注释掉
            continue  # 跳过本次处理

        alpha_powers = []
        for idx, channel in enumerate(eeg_channels):
            # 提取单个通道的数据
            channel_data = eeg_data[idx, :]

            # 确保 channel_data 是一维 NumPy 数组
            if not isinstance(channel_data, np.ndarray) or channel_data.ndim != 1:
                # logging.error(f"通道 {channel} 的数据不是一维 NumPy 数组，类型为: {type(channel_data)}")  # 移除或注释掉
                continue  # 跳过该通道

            # 去除眼动伪迹
            denoised_data = remove_artifacts_with_wavelet(channel_data)

            # 计算 Welch 的功率谱密度 (PSD)
            try:
                psd = DataFilter.get_psd_welch(
                    denoised_data,
                    nfft=256,
                    overlap=128,
                    sampling_rate=sampling_rate,
                    window=WindowOperations.HANNING.value
                )
            except Exception as e:
                logging.error(f"通道 {channel} 的 PSD 计算失败: {e}")
                continue  # 如果 PSD 计算失败，跳过该通道

            # 提取 Alpha 波功率
            try:
                alpha_power = DataFilter.get_band_power(psd, ALPHA_BAND[0], ALPHA_BAND[1])
                alpha_powers.append(alpha_power)
            except Exception as e:
                logging.error(f"通道 {channel} 的 Alpha 波功率提取失败: {e}")

        if not alpha_powers:
            # logging.warning("没有可用的 Alpha 波功率数据来计算平均值。")  # 移除或注释掉
            continue  # 如果没有 Alpha 波功率数据，跳过本次循环

        # 计算所有通道的平均 Alpha 波功率
        avg_alpha_power = np.mean(alpha_powers)
        # 使用 print 输出平均 Alpha 波功率，格式化时间戳到毫秒
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} INFO:平均 Alpha 波功率: {avg_alpha_power:.2f}")

        # 根据 Alpha 波功率阈值控制 LED
        if avg_alpha_power > 10:  # 根据需要调整阈值
            send_led_command(1)  # 打开 LED
        else:
            send_led_command(0)  # 关闭 LED

        # 计算等待时间以防数据重叠
        wait_time = 256 / sampling_rate
        time.sleep(wait_time)

except KeyboardInterrupt:
    # logging.info("用户中断，停止数据流。")  # 移除或注释掉
    pass

except Exception as e:
    logging.error(f"发生意外错误: {e}")

finally:
    # 确保正确清理资源
    try:
        board.stop_stream()
        # logging.info("BrainFlow 数据流已停止。")  # 移除或注释掉
    except Exception as e:
        logging.error(f"停止 BrainFlow 数据流时出错: {e}")
    
    try:
        board.release_session()
        # logging.info("BrainFlow 会话已释放。")  # 移除或注释掉
    except Exception as e:
        logging.error(f"释放 BrainFlow 会话时出错: {e}")
    
    if 'arduino' in locals() and arduino.is_open:
        try:
            arduino.close()
            # logging.info("Arduino 串口连接已关闭。")  # 移除或注释掉
        except Exception as e:
            logging.error(f"关闭 Arduino 串口连接时出错: {e}")
