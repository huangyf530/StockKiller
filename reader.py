import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
import dateutil
from utils import *

DATASET = "/Users/huangyf/Dataset/PRData"

class Reader:
    def __init__(self, data_path, dt=10, a=900, b=3600, k=0.2):
        self.tick_path = data_path + os.sep + "Tick"
        self.order_path = data_path + os.sep + "Order"
        self.orderqueue_path = data_path + os.sep + "OrderQueue"
        self.dt = float(dt)
        self.a = a
        self.b = b
        self.k = k
    
    def read_tick(self):
        self.tickfiles = os.listdir(self.tick_path)
        self.tickfiles = sorted(self.tickfiles)
        for filename in self.tickfiles:
            print(filename)
            df = pd.read_csv(self.tick_path + os.sep + filename)
            df['nTime'] = pd.to_datetime(df.nTime,format='%H%M%S%f')
            df['nPrice'] = df['nPrice'] / 10000    # change its unit
            
            time, prices = self.getPrices(df)
            plotByTime(time, prices)
            exit(0)

    def calPrice(self, turover1, turover2, volume1, volume2, priceB, priceA):
        price = self.k * (float(turover1 - turover2)) / (volume1 - volume2) + (1 - self.k) * float(priceA + priceB) / 20000
        return price
    
    def getPrices(self, df):
        prices = []
        prices.append(df['nPreClose'][0] / 10000)
        x = []
        x.append(pd.Timestamp("1900-01-01 9:29:50"))
        time = pd.Timestamp("1900-01-01 9:29:50")
        endTime = pd.Timestamp("1900-01-01 15:00:00")
        i = 1
        while (time - endTime).total_seconds() != 0:
            x.append(time + dateutil.relativedelta.relativedelta(seconds=10))
            time = time + dateutil.relativedelta.relativedelta(seconds=10)
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
        time = times[0]
        endTime = times[-1]
        data = [first]
        while(i < len(df['nTime']) and (time - endTime).total_seconds() != 0):
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
            time = time + dateutil.relativedelta.relativedelta(seconds=10)
        return data
    
    def getListDataByTime(self, times, dataname, df, first=0):
        i = 1
        time = times[0]
        endTime = times[-1]
        data = [first]
        while(i < len(df['nTime']) and (time - endTime).total_seconds() != 0):
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
            time = time + dateutil.relativedelta.relativedelta(seconds=10)
        return data

if __name__=="__main__":
    reader = Reader(DATASET)
    reader.read_tick()