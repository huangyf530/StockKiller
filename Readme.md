# Big homework for Parttern Recognition

Created by Yufei Huang and Jaewon Kim

## 股市术语
+ Turover：成交额
+ Volume：成交量
+ Bid Price：买家出的价
+ Ask Price：卖家出的价


## 环境配置

+ Python3

+ pytorch
+ sklearn
+ matplotlib
+ pandas

## 参数设置

模型的参数设置在`train.py`文件的15行到38行，可以通过修改args对应的数据进行修改

```python
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
args['path'] = './PRdata'
```

## 运行方法

将数据文件`PRData`放置在指定目录，`PRData`中应包含至少三个文件夹`Tick`、`Order`、`OrderQueue`，运行程序后会在改数据目录中新建目录`HandleTick`存放已经预处理好的数据。

```shell
python3 train.py
```

> 注：如果采用GPU，注意需要GPU内存至少2G

