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

def writeToCsv(path, time, data):
    with open(path, 'w') as f:
        to_write = pd.DataFrame({"nTime" : time, "Data" : data}, columns=['nTime', 'Data'])
        to_write['nTime'].dt.strftime('%H%M%S%f')
        print(to_write.to_csv(index=False), file=f)