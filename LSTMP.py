import torch

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from distutils.util import strtobool
import math
import json
from torch.nn import Parameter
from torch.nn import ParameterList
#from torch.nn.parameter import Parameter
# uncomment below if you want to use SRU
# and you need to install SRU: pip install sru[cuda].
# or you can install it from source code: https://github.com/taolei87/sru.
# import sru

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def act_fun(act_type):

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!

class LSTMP(nn.Module):
    def __init__(self, options, inp_dim):
        super(LSTMP, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        #print('input= ', self.input_dim)
        self.lstm_lay = list(map(int, options["lstm_lay"].split(",")))
        self.proj_lay =  list(map(int, options["proj_lay"].split(",")))
        self.lstm_drop = list(map(float, options["lstm_drop"].split(",")))
        self.lstm_use_batchnorm = list(map(strtobool, options["lstm_use_batchnorm"].split(",")))
        self.lstm_use_laynorm = list(map(strtobool, options["lstm_use_laynorm"].split(",")))
        self.lstm_use_laynorm_inp = strtobool(options["lstm_use_laynorm_inp"])
        self.lstm_use_batchnorm_inp = strtobool(options["lstm_use_batchnorm_inp"])
        self.lstm_act = options["lstm_act"].split(",")
        self.lstm_orthinit = strtobool(options["lstm_orthinit"])

        self.bidir = strtobool(options["lstm_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wfx = nn.ModuleList([])  # Forget
        self.ufr = nn.ModuleList([])  # Forget

        self.wix = nn.ModuleList([])  # Input
        self.uir = nn.ModuleList([])  # Input

        self.wox = nn.ModuleList([])  # Output
        self.uor = nn.ModuleList([])  # Output

        self.wcx = nn.ModuleList([])  # Cell state
        self.ucr = nn.ModuleList([])  # Cell state

        self.urh = nn.ModuleList([])  #projection layer

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wfx = nn.ModuleList([])  # Batch Norm
        self.bn_wix = nn.ModuleList([])  # Batch Norm
        self.bn_wox = nn.ModuleList([])  # Batch Norm
        self.bn_wcx = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.lstm_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.lstm_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_lstm_lay = len(self.lstm_lay)

        current_input = self.input_dim
        #print(current_input) 

        # Initialization of hidden layers
        for i in range(self.N_lstm_lay):

            # Activations
            self.act.append(act_fun(self.lstm_act[i]))

            add_bias = True

            if self.lstm_use_laynorm[i] or self.lstm_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))


            # Recurrent connections
            self.ufr.append(nn.Linear(self.proj_lay[i], self.lstm_lay[i], bias=False))
            self.uir.append(nn.Linear(self.proj_lay[i], self.lstm_lay[i], bias=False))
            self.uor.append(nn.Linear(self.proj_lay[i], self.lstm_lay[i], bias=False))
            self.ucr.append(nn.Linear(self.proj_lay[i], self.lstm_lay[i], bias=False))
            self.urh.append(nn.Linear(self.lstm_lay[i], self.proj_lay[i], bias=False))

            if self.lstm_orthinit:
                nn.init.orthogonal_(self.ufr[i].weight)
                nn.init.orthogonal_(self.uir[i].weight)
                nn.init.orthogonal_(self.uor[i].weight)
                nn.init.orthogonal_(self.ucr[i].weight)
                nn.init.orthogonal_(self.urh[i].weight)


            # batch norm initialization
            self.bn_wfx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wix.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wox.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wcx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.lstm_lay[i]))

            if self.bidir:
                current_input = 2 * self.proj_lay[i]
            else:
                current_input = self.proj_lay[i]

        self.out_dim = self.proj_lay[i] + self.bidir * self.proj_lay[i]
    def forward(self, x):
    
            # Applying Layer/Batch Norm
            if bool(self.lstm_use_laynorm_inp):
                x = self.ln0((x))
    
            if bool(self.lstm_use_batchnorm_inp):
                x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
                x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
    
            for i in range(self.N_lstm_lay):
    
                # Initial state and concatenation
                if self.bidir:
                    r_init = torch.zeros(2 * x.shape[1], self.proj_lay[i])
                    x = torch.cat([x, flip(x, 0)], 1)
                else:
                    c_init = torch.zeros(x.shape[1], self.lstm_lay[i])
                    r_init = torch.zeros(x.shape[1], self.proj_lay[i])
    
                # Drop mask initilization (same mask for all time steps)
                if self.test_flag == False:
                    drop_mask = torch.bernoulli(torch.Tensor(c_init.shape[0], c_init.shape[1]).fill_(1 - self.lstm_drop[i]))
                else:
                    drop_mask = torch.FloatTensor([1 - self.lstm_drop[i]])
    
                if self.use_cuda:
                    c_init = c_init.cuda()
                    r_init = r_init.cuda()
                    drop_mask = drop_mask.cuda()
    
                # Feed-forward affine transformations (all steps in parallel)
                wfx_out = self.wfx[i](x)
                wix_out = self.wix[i](x)
                wox_out = self.wox[i](x)
                wcx_out = self.wcx[i](x)
    
                # Apply batch norm if needed (all steos in parallel)
                if self.lstm_use_batchnorm[i]:
    
                    wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] * wfx_out.shape[1], wfx_out.shape[2]))
                    wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1], wfx_out.shape[2])
    
                    wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] * wix_out.shape[1], wix_out.shape[2]))
                    wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1], wix_out.shape[2])
    
                    wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] * wox_out.shape[1], wox_out.shape[2]))
                    wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1], wox_out.shape[2])
    
                    wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] * wcx_out.shape[1], wcx_out.shape[2]))
                    wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1], wcx_out.shape[2])
    
                # Processing time steps
                hiddens = []
                ct = c_init
                ht = c_init
                rt = r_init
                #print(ct.size())
                #print(rt.size())
                for k in range(x.shape[0]):
                    #print(k) 
                    # LSTMP equations
                    #print('rt=',rt.size())
                    #print('fx=',wfx_out[k].size())
                    ft = torch.sigmoid(wfx_out[k] + self.ufr[i](rt))
                    #print('ft=',ft.size())
                    #print('ix=',wix_out[k].size())
                    it = torch.sigmoid(wix_out[k] + self.uir[i](rt))
                    #print('it=',it.size())
                    #print('ox=',wox_out[k].size())
                    ot = torch.sigmoid(wox_out[k] + self.uor[i](rt))
                    #print('ot=',ot.size())
                    #print('ct1=',ct.size())
                    #print('cx=',wcx_out[k].size())
                    #a=self.ucr[i](rt)
                    #print('a=',a.size())
                    #print('drop_mask=',drop_mask.size())
                    ct = it * self.act[i](wcx_out[k] + self.ucr[i](rt)) * drop_mask + ft * ct
                    #print('ct=',ct.size)
                    ht = ot * self.act[i](ct)
                    #print('ht=',ht.size())
                    rt = self.urh[i](ht)
    
    
                    if self.lstm_use_laynorm[i]:
                       
                        rt = self.ln[i](rt)
    
                    hiddens.append(rt)
    
                # Stacking hidden states
                
                r = torch.stack(hiddens)
    
                # Bidirectional concatenations
                if self.bidir:
                    r_f = r[:, 0 : int(x.shape[1] / 2)]
                    r_b = flip(r[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                    r = torch.cat([r_f, r_b], 2)
    
                # Setup x for the next hidden layer
                x = r
    
            return x
