
from pyOpenBCI import OpenBCICyton

def print_raw(sample):
    print(sample.channels_data)

board = OpenBCICyton(port='COM3', daisy=False)

board.start_stream(print_raw)


#这么运行 py -3.8 BCI.py 
#需要在低版本python环境下运行