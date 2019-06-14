import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PRNet(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers):
        super(PRNet, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=True,
                            bidirectional=False,
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=True,
                            bidirectional=False,
                            batch_first=True)
        self.logits = nn.Linear(in_features=hidden_size,
                                out_features=1,
                                bias=True)
    
    def forward(self, in_data):
        output = self.lstm1(in_data)
        output = self.lstm2(output)
        output = self.logits(output)
        return output