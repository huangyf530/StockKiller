import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

register_matplotlib_converters()

def plotByTime(time, data):
    # for i in range(len(data)):
    #     print(i, time[i], data[i])
    to_plot = pd.DataFrame({"Time" : time, "Data" : data}, columns=['Time', 'Data'])
    # to_plot.index = to_plot['Time']
    plt.plot(to_plot['Data'])
    plt.show()

def Interpolation(value_shift, x_shift, my_x_shift, left_value):
    y = float(value_shift) / x_shift * my_x_shift + left_value
    return y