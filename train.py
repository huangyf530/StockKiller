import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from reader import Reader
from model import PRNet
import time
import os

isGPU = False

# parameters
args = dict()
args['predict_len'] = 60 # 5min
args['epoch'] = 100
args['learning_rate'] = 0.1
args['batch_size'] = 500
args['lr_decay_factor'] = 0.99
args['input_dim'] = 1
args['hidden_size'] = 50
args['num_layers'] = 2
args['a'] = 30
args['b'] = 300
args['dt'] = 5
args['k'] = 0.3
args['theta'] = 0.004
args['save_path'] = './models/'
args['load_path'] = 'model0.pt'
args['step_size'] = 1000
args['load_model'] = False
args['gpu'] = 'cuda:3'


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

    st, ed = 0, min(args['batch_size'], len(data))
    total_loss = 0.
    step = 0
    
    while st < len(data):
        step += 1
        batch_data = torch.FloatTensor(np.array(data[st: ed], np.float32))
        batch_label = torch.FloatTensor(np.array(label[st: ed], np.float32))

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
        
        if step % args['step_size'] == 0:
            print('Step [%d] loss [%.4f]' % (step, float(loss)))
        
        st, ed = ed, min(ed + args['batch_size'], len(data))

    return total_loss / step


def predict(data):
    global device
    global model
    global reader
    
    theta = args['theta']
    pl = args['predict_len']
    ml = len(data[0])
    assert(ml > pl)
    st, ed = 0, min(args['batch_size'], len(data))
    labels = list()
    for i in range(len(data)):
        l, _ = reader.calUpAndDown(data[i], theta)
        labels.append(l)
        
    predict_data = [data[i][: pl] for i in range(len(data))]
    assert (len(predict_data) == len(data))
    while st < len(predict_data):
        for i in range(ml - pl):
            batch_data = [predict_data[t][i: i + pl] for t in range(st, ed)]
            batch_data = torch.FloatTensor(np.array(batch_data, np.float32))

            batch_data = batch_data.to(device)
            
            output = model(batch_data)
            output = output.detach().cpu().numpy().tolist()
            for j in range(len(output)):
                predict_data[st + j].append(output[j])

        st, ed = ed, min(ed + args['batch_size'], len(predict_data))


    predict_labels = list()
    for i in range(len(predict_data)):
        l, _ = reader.calUpAndDown(predict_data[i], theta)
        predict_labels.append(l)

    assert(len(labels) == len(predict_labels))

    total_num = 0
    acc_num = 0
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            total_num += 1
            if labels[i][j] == predict_labels[i][j]:
                acc_num += 1

    return float(acc_num) / float(total_num)


# init data
path = './PRData'
reader = Reader(path, args)
stock_time, all_price, _  = reader.read_tick()
tv_price, test_price, _, _ = train_test_split(all_price, [i for i in range(len(all_price))], test_size=0.1, random_state=42)
train_price, valid_price, _, _ = train_test_split(tv_price, [i for i in range(len(tv_price))], test_size=0.1, random_state=42)

train_data, train_labels = turn2data(train_price)
valid_data, valid_labels = turn2data(valid_price)
test_data, test_labels = turn2data(test_price)

isGPU = torch.cuda.is_available()
# create model
if isGPU:
    print('GPU environment')
    device = torch.device(args['gpu'])
else:
    print('CPU environment')
    device = torch.device('cpu')

model = PRNet(args)
if args['load_model']:
     if isGPU:
        model.load_state_dict(torch.load(args['save_path'] + args['load_path'],
                                         map_location=args['gpu']))
     else:
        model.load_state_dict(torch.load(args['save_path'] + args['load_path'],
                                         map_location=device))
    
model.to(device)

if not os.path.exists(args['save_path']):
    os.makedirs(args['save_path'])
        
optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
loss_func = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

for ep in range(args['epoch']):
    st = time.time()
    
    data, label = shuffle_data(train_data, train_labels)

    # train
    model.train()
    loss = train(data, label)
    #pacc = predict(train_price)
    print('Epoch %d: learning rate %.8f epoch time %.4fs mean loss [%.4f]'
          % (ep, get_lr(optimizer), time.time()-st, loss))

    torch.save(model.state_dict(), args['save_path'] + 'model' + str(ep) + '.pt')

    # valid
    model.eval()
    loss = train(valid_data, valid_labels, False)
    scheduler.step()
    pacc = predict(valid_price)
    print('         validation_set, loss [%.4f] label acc [%.4f]'
          % (loss, pacc))
    
    # test
    loss = train(test_data, test_labels, False)
    pacc = predict(test_price)
    print('         test_set, loss [%.4f] label acc [%.4f]'
          % (loss, pacc))
