import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from reader import Reader
from model import PRNet
import time
import os
import math
from utils import *

isGPU = False

# parameters
args = dict()
args['predict_len'] = 120 # 10min
args['epoch'] = 100
args['learning_rate'] = 0.0001
args['batch_size'] = 2000
args['lr_decay_factor'] = 0.9
args['input_dim'] = 1
args['hidden_size'] = 100
args['num_layers'] = 2
args['a'] = 30
args['b'] = 300
args['dt'] = 5
args['k'] = 0.3
args['theta'] = 0.004
args['save_path'] = './models_'+'pl'+str(args['predict_len'])+'_lr'+str(args['learning_rate'])+'_hd'+str(args['hidden_size'])
args['load_path'] = 'model6.pt'
args['step_size'] = 1000
args['load_model'] = True
args['gpu'] = 'cuda:3'
args['isTrain'] = False
args['imagepath'] = "./Image_" + 'pl'+str(args['predict_len'])+'_lr'+str(args['learning_rate'])+'_hd'+str(args['hidden_size'])


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

    if args['imagepath'] is not None and not args['isTrain']:
        if not os.path.exists(args['imagepath']):
            os.mkdir(args['imagepath'])
        for i in range(len(data)):
            plotPredictAndPrice(data[i],  predict_data[i], pl, os.path.join(args['imagepath'], "figure" +  str(i) + ".png"))
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

    acc_rate, call_rate = calPandR(predict_labels, labels)   
    return float(acc_num) / float(total_num), acc_rate, call_rate


def newpredict(data):
    global device
    global model
    global reader

    theta = args['theta']
    pl = args['predict_len']
    ml = len(data[0])
    a_index = math.ceil(args['a'] / args['dt'])
    b_index = math.floor(args['b'] / args['dt'])

    st, ed = 0, min(args['batch_size'], len(data))

    labels = list()
    for i in range(len(data)):
        l, _ = reader.calUpAndDown(data[i], theta)
        labels.append(l)

    predict_data = [list() for _ in range(len(data))]
    predict_labels = [list() for _ in range(len(data))]
    
    while st < len(data):
        for i in range(pl, ml - b_index):
            temp_data = [data[se][i - pl: i] for se in range(st, ed)]
            for f in range(pl, pl + b_index + 1):
                batch_data = [temp_data[t][f - pl: f] for t in range(st, ed)]
                batch_data = torch.FloatTensor(np.array(batch_data, np.float32))
                batch_data = batch_data.to(device)
                
                output = model(batch_data)
                output = output.detach().cpu().numpy().tolist()

                if f == i:
                    for j in range(len(output)):
                        predict_data[st + j].append(output[j])
                        
                for j in range(len(output)):
                    temp_data[j].append(output[j])

            tt = [temp_data[se][pl:] for se in range(st, ed)]
            classify_res = reader.getClassify(tt, theta)
            for j in range(len(classify_res)):
                predict_labels[st + j].append(classify_res[j])

        st, ed = ed, min(ed + args['batch_size'], len(data))


    for i in range(len(data)):
        predict_data[i] = data[i][: pl] + predict_data[i]

    '''
    if args['imagepath'] is not None and not args['isTrain']:
        if not os.path.exists(args['imagepath']):
            os.mkdir(args['imagepath'])
        for i in range(len(data)):
            plotPredictAndPrice(data[i],  predict_data[i], pl, os.path.join(args['imagepath'], "figure" +  str(i) + ".png"))
    '''

    cmplabels = [list() for _ in range(len(labels))]
    elen = len(predict_labels[0])
    for i in range(len(labels)):
        cmplabels[i] = labels[pl: pl + elen]

    assert(len(cmplabels) == len(predict_labels))
    total_num = 0
    acc_num = 0
    
    for i in range(len(predict_labels)):
        assert(len(cmplabels[i]) == len(predict_labels[i]))
        for j in range(len(predict_labels[i])):
            total_num += 1
            if cmplabels[i][j] == predict_labels[i][j]:
                acc_num += 1

    acc_rate, call_rate = calPandR(predict_labels, cmplabels)
    return float(acc_num) / float(total_num), acc_rate, call_rate
            
# init data
path = './PRdata'
reader = Reader(path, args)
stock_time, all_price, _  = reader.read_tick()
tv_price, test_price, _, _ = train_test_split(all_price, [i for i in range(len(all_price))], test_size=0.1, random_state=42)
train_price, valid_price, _, _ = train_test_split(tv_price, [i for i in range(len(tv_price))], test_size=0.1, random_state=42)
# cal three rate
# down_train, nochange_train, up_train = getThreeRate(train_price, reader, args['theta'])
# down_valid, nochange_valid, up_valid = getThreeRate(valid_price, reader, args['theta'])
# down_test, nochange_test, up_test = getThreeRate(test_price, reader, args['theta'])
# print("In train set: down: [%.4f], no change: [%.4f], up rate: [%.4f]" % (down_train, nochange_train, up_train))
# print("In valid set: down: [%.4f], no change: [%.4f], up rate: [%.4f]" % (down_valid, nochange_valid, up_valid))
# print("In test  set: down: [%.4f], no change: [%.4f], up rate: [%.4f]" % (down_test, nochange_test, up_test))
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
        model.load_state_dict(torch.load(args['save_path'] + os.sep + args['load_path'],
                                         map_location=args['gpu']))
     else:
        model.load_state_dict(torch.load(args['save_path'] + os.sep + args['load_path'],
                                         map_location=device))
    
model.to(device)

if not os.path.exists(args['save_path']):
    os.makedirs(args['save_path'])
        
optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
loss_func = nn.MSELoss()
train_loss_list = list()
valid_loss_list = list()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=args['lr_decay_factor'])

for ep in range(args['epoch']):
    st = time.time()
    
    data, label = shuffle_data(train_data, train_labels)

    # train
    if args['isTrain']:
        model.train()
        loss = train(data, label)
        train_loss_list.append(loss)
        #pacc = predict(train_price)
        print('Epoch %d: learning rate %.8f epoch time %.4fs mean loss [%.4f]'
            % (ep, get_lr(optimizer), time.time()-st, loss))

        torch.save(model.state_dict(), args['save_path'] + '/model' + str(ep) + '.pt')

    # valid
    model.eval()
    loss = train(valid_data, valid_labels, False)
    valid_loss_list.append(loss)
    scheduler.step()
    pacc, acc_rate, call_rate = newpredict(valid_price)
    print('         validation_set, loss [%.4f] label acc [%.4f] accurate [%.4f] recall [%.4f]'
          % (loss, pacc, acc_rate, call_rate))
    
    # test
    loss = train(test_data, test_labels, False)
    pacc, acc_rate, call_rate = newpredict(test_price)
    print('         test_set, loss [%.4f] label acc [%.4f] accurate [%.4f] recall [%.4f]'
          % (loss, pacc, acc_rate, call_rate))
    if not args['isTrain']:
        break
    elif args['imagepath'] is not None:
        if not os.path.exists(args['imagepath']):
            os.mkdir(args['imagepath'])
        plotLoss(train_loss_list, valid_loss_list, os.path.join(args['imagepath'], "loss.png"))
