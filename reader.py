import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
import dateutil
from utils import *
import torch.utils.data as data
import math

DATASET = "/Users/huangyf/Dataset/PRData"

class Reader(data.Dataset):
    def __init__(self, data_path, args):
        self.tick_path = data_path + os.sep + "Tick"
        self.order_path = data_path + os.sep + "Order"
        self.orderqueue_path = data_path + os.sep + "OrderQueue"
        self.handled_data = data_path + os.sep + "HandleTick"
        self.datapath = data_path
        self.dt = float(args['dt'])
        self.a = args['a']
        self.b = args['b']
        self.k = args['k']
    
    def read_tick(self):
        all_prices = []
        if os.path.exists(os.path.join(self.datapath, "abandon_file")):
            with open(os.path.join(self.datapath, "abandon_file"), 'r') as f:
                line = f.readline()
                line = line.strip()
                abandon_file = line.split(" ")
        else:
            abandon_file = []
        abandon = open(os.path.join(self.datapath, "abandon_file"), 'a')
        if not os.path.exists(self.handled_data):
            os.mkdir(self.handled_data)
        self.datafiles = os.listdir(self.handled_data)
        self.datafiles = sorted(self.datafiles)
        time = self.getTime()
        if not os.path.exists(self.tick_path):
            # just handled files 
            print("Load Price Data from handled files")
            print(abandon_file, "is abandoned")
            for filename in self.datafiles:
                df = pd.read_csv(self.handled_data + os.sep + filename)
                df['nTime'] = pd.to_datetime(df.nTime,format='%Y-%m-%d %H:%M:%S')
                all_prices.append(df['Data'].tolist())
                continue
            abandon.close()
            self.time = time[1:]
            print("Load Over! Valid file number is", len(all_prices), "Abandon file number is", len(abandon_file))
            return time[1:], all_prices, abandon_file
        print("Get data from Tick and HandleTick")
        self.tickfiles = os.listdir(self.tick_path)
        self.tickfiles = sorted(self.tickfiles)
        # k = 0
        for filename in self.tickfiles:
            # k += 1
            # if k == 10:
            #     break
            # print(filename)
            if filename in self.datafiles:
                # print("read {} from handled file".format(filename))
                df = pd.read_csv(self.handled_data + os.sep + filename)
                df['nTime'] = pd.to_datetime(df.nTime,format='%Y-%m-%d %H:%M:%S')
                all_prices.append(df['Data'].tolist())
                continue
            if filename in abandon_file:
                print(filename, "is in abandon list")
                continue
            df = pd.read_csv(self.tick_path + os.sep + filename)
            df['nTime'] = pd.to_datetime(df.nTime,format='%H%M%S%f')
            df['nPrice'] = df['nPrice'] / 10000    # change its unit
            if df.size == 0:
                abandon_file.append(filename)
                abandon.write(filename + " ")
                print(filename, "is abandoned because df.size == 0")
                continue
            if df['nTime'][0].hour != 9:
                abandon_file.append(filename)
                abandon.write(filename + " ")
                print(filename, "is abandoned because df['nTime'][0].hour != 9")
                continue
            if df['nTime'].iloc[-1].hour < 14 and df['nTime'].iloc[-1].minute < 59:
                abandon_file.append(filename)
                abandon.write(filename + " ")
                print(filename, "is abandoned because df['nTime'].iloc[-1].hour < 14")
                continue
            time, prices = self.getPrices(df)
            all_prices.append(prices[1:])
            writeToCsv(self.handled_data + os.sep + filename, time[1:], prices[1:])
            # plotByTime(time[1:], prices[1:])
        abandon.close()
        self.time = time[1:]
        print("Load Over! Valid file number is", len(all_prices), "Abandon file number is", len(abandon_file))
        return time[1:], all_prices, abandon_file

    def calPrice(self, turover1, turover2, volume1, volume2, priceB, priceA):
        price = self.k * (float(turover1 - turover2)) / (volume1 - volume2) + (1 - self.k) * float(priceA + priceB) / 20000
        return price

    def getTime(self):
        x = []
        x.append(pd.Timestamp("1900-01-01 9:29:55"))
        time = pd.Timestamp("1900-01-01 9:29:55")
        endTime1 = pd.Timestamp("1900-01-01 11:30:00")
        beginTime = pd.Timestamp("1900-01-01 13:00:00")
        endTime2 = pd.Timestamp("1900-01-01 15:00:00")
        i = 1
        while (time - endTime1).total_seconds() < 0:
            x.append(time + dateutil.relativedelta.relativedelta(seconds=self.dt))
            time = time + dateutil.relativedelta.relativedelta(seconds=self.dt)
        time = beginTime
        x.append(beginTime)
        while (time - endTime2).total_seconds() < 0:
            x.append(time + dateutil.relativedelta.relativedelta(seconds=self.dt))
            time = time + dateutil.relativedelta.relativedelta(seconds=self.dt)
        return x
    
    def getPrices(self, df):
        prices = []
        prices.append(df['nPreClose'][0] / 10000)
        x = self.getTime()
        turover = self.getIntDataByTime(x, 'iAccTurover', df)
        volume = self.getIntDataByTime(x, 'iAccVolume', df)
        bid_price = self.getListDataByTime(x, 'nBidPrice', df)
        ask_price = self.getListDataByTime(x, 'nAskPrice', df)
        for i in range(1, len(x)):
            price = 0
            if volume[i] == volume[i - 1]:
                price = prices[-1]
            else:
                price = self.calPrice(turover[i], turover[i - 1], volume[i], volume[i - 1], bid_price[i], ask_price[i])
            prices.append(price)
        return x, prices
    
    def getIntDataByTime(self, times, dataname, df, first=0):
        i = 1
        time_index = 1
        time = times[1]
        data = [first]
        while(i < len(df['nTime']) and time_index < len(times)):
            time = times[time_index]
            delta = (df['nTime'][i] - time).total_seconds()
            if delta < 0:
                i += 1
                continue
            current_data = 0
            if delta == 0:
                current_data = df[dataname][i]
            if delta > 0:
                x_shift = (df['nTime'][i] - df['nTime'][i - 1]).total_seconds()
                my_x_shift = (time - df['nTime'][i - 1]).total_seconds()
                current_data = Interpolation(df[dataname][i] - df[dataname][i - 1], x_shift, my_x_shift, df[dataname][i - 1])
            data.append(current_data)
            time_index += 1
        while time_index < len(times):
            time = times[time_index]
            current_data = df[dataname][i-1]
            data.append(current_data)
            time_index += 1
        return data
    
    def getListDataByTime(self, times, dataname, df, first=0):
        i = 1
        time = times[1]
        endTime = times[-1]
        data = [first]
        time_index = 1
        while(i < len(df['nTime']) and time_index < len(times)):
            time = times[time_index]
            delta = (df['nTime'][i] - time).total_seconds()
            if delta < 0:
                i += 1
                continue
            current_data = 0
            if delta == 0:
                current_list = df[dataname][i][1:-1].split(",")
                current_data = float(current_list[0])
            if delta > 0:
                x_shift = (df['nTime'][i] - df['nTime'][i - 1]).total_seconds()
                my_x_shift = (time - df['nTime'][i - 1]).total_seconds()
                current_list = df[dataname][i][1:-1].split(",")
                last_list = df[dataname][i - 1][1:-1].split(",")
                current_data = Interpolation(float(current_list[0]) - float(last_list[0]), x_shift, my_x_shift, float(last_list[0]))
            data.append(current_data)
            time_index += 1
        while time_index < len(times):
            current_list = df[dataname][i-1][1:-1].split(",")
            current_data = float(current_list[0])
            data.append(current_data)
            time_index += 1
        return data
    
    def calUpAndDown(self, price, theta):
        '''
        计算每个时间点股票价格属于上涨还是降低或者是不变
        @args:
            price: 价格列表，记录某一天每个时刻的价格
            theta: 阈值
        @return: 
            result: type:list, -1=down, 0=no change, 1=up
            time: 对应的时间戳
        '''
        a_index = math.ceil(self.a / self.dt)
        b_index = math.floor(self.b / self.dt)
        result = []
        time = []
        for i in range(len(price) - b_index):
            max_gap = 0
            max_index = 0
            for j in range(a_index, b_index + 1):
                if(abs(price[i + j] - price[i]) > max_gap):
                    max_gap = abs(price[i + j] - price[i])
                    max_index = i + j
            d = float(price[max_index] - price[i]) / price[i]
            if d < -theta:
                result.append(-1)
            elif d > theta:
                result.append(1)
            else:
                result.append(0)
            time.append(self.time[i])
        return result, time
    
    def  getClassify(self, price, theta):
        '''
        计算某个时间点股票价格属于上涨还是降低或者是不变
        @args:
            price: (size = N, *)价格列表，记录某一天每个时刻的价格
            theta: 阈值
        @return: 
            result: type:list(size = N), -1=down, 0=no change, 1=up
        '''
        a_index = math.ceil(self.a / self.dt)
        b_index = math.floor(self.b / self.dt)
        result = []
        for i in range(len(price)):
            max_gap = 0
            max_index = 0
            for j in range(a_index, b_index + 1):
                if(abs(price[i][j] - price[i][0]) > max_gap):
                    max_gap = abs(price[i][j] - price[i][0])
                    max_index = j
            d = float(price[i][max_index] - price[i][0]) / price[i][0]
            if d < -theta:
                result.append(-1)
            elif d > theta:
                result.append(1)
            else:
                result.append(0)
        return result

if __name__=="__main__":
    # args = dict()
    # args['a'] = 30
    # args['b'] = 300
    # args['dt'] = 5
    # args['k'] = 0.3
    # args['theta'] = 0.004
    # reader = Reader(DATASET, args)
    # reader.read_tick()
    y1 = np.random.randn(10)
    y2 = np.random.randn(10)
    y3 = np.random.randn(10)
    y4 = np.random.randn(10)
    plotPredictAndPrice(y1, y2, 2, "./figure1.png")
    plotPredictAndPrice(y3, y4, 2, "./figure2.png")