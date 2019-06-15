import torch
import torch.nn as nn
import torch.nn.functional as F

class PRNet(nn.Module):
    def __init__(self, args):
        super(PRNet, self).__init__()
        self.input_dim = args['input_dim']
        self.batch_size = args['batch_size']
        self.max_length = args['predict_len']
        self.hidden_size = args['hidden_size']
        self.num_layers = args['num_layers']
        
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=True,
                            dropout=0.1
        )

        self.fc1 = nn.Linear(in_features=self.hidden_size,
                             out_features=1
        )

        
    def forward(self, x):
        # x: [batch, len]
        # x: [len, batch, dim]
        x = x.permute(1, 0)
        x = x.view(self.max_length, -1, self.input_dim)
        # output: [len, batch, num_direction * hidden_dim]
        output, _ = self.lstm(x)
        last_out = output[-1]
        res = self.fc1(last_out)
        res = res.view(-1)
        return res
