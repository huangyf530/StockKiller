import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from reader import Reader
from model import PRNet
import time

isGPU = False

# parameters
args = dict()
args['predict_len'] = 60 # 5min
args['epoch'] = 100
args['learning_rate'] = 0.0001
args['batch_size'] = 50
args['lr_decay_factor'] = 0.99
args['hidden_size'] = 200
args['num_layers'] = 2

# func
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def shuffle_data(x, y):
    shuffled_range = list(range(len(x)))
    _x = list()
    _y = list()
    np.random.shuffle(shuffled_range)
    for i in range(len(x)):
        _x.append(x[shuffled_range[i]])
        _y.append(y[shuffled_range[i]])
    return _x, _y

def turn2data(dayprices):
    data = list()
    labels = list()
    pl = args['predict_len']
    for dp in dayprices:
        for i in range(len(dp) - pl):
            data.append(dp[i: i+pl])
            labels.append(dp[i+pl])
    return data, labels

# init data
path = './PRData'
reader = Reader(path)
all_price, stock_time = reader.read_tick()

tv_price, test_price, _, _ = train_test_split(all_price, [i for i in range(len(all_price))], test_size=0.1, random_state=42)
train_price, valid_price, _, _ = train_test_split(tv_price, [i for i in range(len(tv_price))], test_size=0.1, random_state=42)

train_data, train_labels = turn2data(train_price)
valid_data, valid_labels = turn2data(valid_price)
test_data, test_labels = turn2data(test_price)

isGPU = torch.cuda.is_available()


