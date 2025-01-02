import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import numpy as np
import serial

# 选择Board ID
# 对于OpenBCI Cyton，通常使用自定义Board ID
# 你可以在BrainFlow的board list中找到相应ID
# 例如，假设使用BOARD_OPENBCI_CYTON_BOARD = 0
# 实际使用中请参考BrainFlow文档
BOARD_ID = 0  # 请根据实际情况调整

# 设置日志级别（可选）
BoardShim.enable_dev_board_logger()
# BoardShim.disable_board_logger()

# 配置输入参数
params = BrainFlowInputParams()
params.serial_port = 'COM3'  # 请替换为你的COM端口

def main():
    try:
        # 初始化Board
        board = BoardShim(BOARD_ID, params)
        board.prepare_session()
        board.start_stream()

        print("开始采集数据...")
        time.sleep(10)  # 采集10秒的数据

        # 获取数据
        data = board.get_board_data()
        print("采集到的数据形状：", data.shape)

        # 示例：提取前8个EEG通道的数据
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)[:8]
        eeg_data = data[eeg_channels, :]

        # 打印部分数据
        print("前8个通道的EEG数据（最近10个样本）：")
        print(eeg_data[:, -10:])

        # 可选：保存数据到CSV
        DataFilter.write_file(data, "eeg_data.csv", 'w')
        print("数据已保存到 eeg_data.csv")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 停止并释放资源
        board.stop_stream()
        board.release_session()
        print("采集结束，资源已释放。")

if __name__ == "__main__":
    main()
