import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
import torchvision.models as models

class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, n_hidden, n_LSTM_layers, linear_layer_dims):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_LSTM_layers # number of LSTM layers (stacked)

        print(n_features, seq_length, n_hidden, n_LSTM_layers, linear_layer_dims)
    
        self.l_lstm = torch.nn.LSTM(input_size = self.n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.linear_relus = nn.ModuleList()
        last_dim = self.n_hidden

        for cur_linear_layer_dim in linear_layer_dims:
            self.linear_relus += [torch.nn.Linear(last_dim, cur_linear_layer_dim), torch.nn.LeakyReLU(), nn.Dropout(p=0.15)]
            last_dim = cur_linear_layer_dim

        self.linear_relus += [torch.nn.Linear(last_dim, 1)]
        self.l_sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        lstm_out, (hidden, cell) = self.l_lstm(x)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        # x = hidden.contiguous().view(batch_size,-1)
        
        res = hidden[-1].squeeze()
        
        for i, layer in enumerate(self.linear_relus):
            res = layer(res)

        return self.l_sigmoid(res)