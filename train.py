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
args['input_dim'] = 1
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

def train(data, label, isTrain=True):
    global device
    global model
    global optimizer

    st, ed = 0, args['batch_size']
    total_loss, total_acc = 0., 0.

    while st < len(data):
        batch_data = torch.LongTensor(np.array(data[st: ed], np.float32))
        batch_label = torch.LongTensor(np.array(label[st: ed], np.float32))

        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)

        # forward
        output = model(batch_data)
        loss = loss_func(output, batch_label)
        if isTrain:
            # clear last grad
            optimizer.zero_grad()
            # backward
            loss.backward()
            # optimize
            optimizer.step()

        total_loss += float(loss)
        total_acc += (batch_label == output.detach()).sum()

        st, ed = ed, min(ed + args['batch_size'], len(data))

    return total_loss, total_acc / len(data)

# init data
path = './PRData'
reader = Reader(path)
all_price, stock_time = reader.read_tick()
np_all = np.array(all_price, np.float32)
tv_price, test_price, _, _ = train_test_split(all_price, [i for i in range(len(all_price))], test_size=0.1, random_state=42)
train_price, valid_price, _, _ = train_test_split(tv_price, [i for i in range(len(tv_price))], test_size=0.1, random_state=42)

train_data, train_labels = turn2data(train_price)
valid_data, valid_labels = turn2data(valid_price)
test_data, test_labels = turn2data(test_price)

isGPU = torch.cuda.is_available()
# create model
if isGPU:
    print('GPU environment')
    device = torch.device('cuda:0')
else:
    print('CPU environment')
    device = torch.device('cpu')

model = PRNet(args)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
loss_func = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args['lr_decay_factor'], patience=3)

for ep in range(args['epoch']):
    st = time.time()

    data, label = shuffle_data(train_data, train_labels)

    # train
    loss, acc = train(data, label)

    print('Epoch %d: learning rate %.8f epoch_time %.4fs loss [%.4f] accuracy [%.4f]'
          % (ep, get_lr(optimizer), time.time()-st, loss, acc))

    # valid
    loss, acc = train(valid_data, valid_labels, False)

    print('         validation_set, loss [%.4f] accuracy [%.4f]'
          % (loss, acc))

    # test
    loss, acc = train(test_data, test_labels, False)
    print('         test_set, loss [%.4f] accuracy [%.4f]'
          % (loss, acc))
