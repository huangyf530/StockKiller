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

def calPandR(predict, label):
    up_total_num = 0
    down_total_num = 0
    predict_up = 0
    predict_down = 0
    correct_up_num = 0
    correct_down_num = 0
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] == 1:
                up_total_num += 1
            if predict[i][j] == 1:
                predict_up += 1
                if label[i][j] == 1:
                    correct_up_num += 1
            if label[i][j] == -1:
                down_total_num += 1
            if predict[i][j] == -1:
                predict_down += 1
                if label[i][j] == -1:
                    correct_down_num += 1
    up_accu = float(correct_up_num) / float(predict_up)
    down_accu = float(correct_down_num) / float(predict_down)
    up_call = float(correct_up_num) / float(up_total_num)
    down_call = float(correct_down_num) / float(down_total_num)
    return (up_accu + down_accu) / 2, (up_call + down_call) / 2
             