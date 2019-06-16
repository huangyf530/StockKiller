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

def plotPredictAndPrice(data, predict, pl, path):
    x1 = range(len(data))
    x2 = range(pl, len(data))
    plt.plot(x1, data)
    plt.plot(x2, predict[pl:])
    plt.savefig(path)

def getThreeRate(data, reader, theta):
    '''
    统计数据中三种类别的占比
    data: 二维列表，第一维表示文件，第二维表示该文件中不同时刻的价格数据
    @Return:
        result[0]: 下降的个数
        result[1]: 不变的个数
        result[2]: 上涨的个数
    '''
    result = [0, 0, 0]
    for i in range(len(data)):
        label, _ = reader.calUpAndDown(data[i], theta)
        for j in range(len(label)):
            result[label[j] + 1] += 1
    sum = result[0] + result[1] + result[2]
    return result[0] / sum, result[1] / sum, result[2] / sum

def plotLoss(train_loss, predict_loss, path):
    x = range(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, predict_loss)
    plt.savefig(path)
