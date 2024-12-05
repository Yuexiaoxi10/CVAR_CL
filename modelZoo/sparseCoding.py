import sys
sys.path.append('../')
sys.path.append('.')

import ipdb
## Imports related to PyTorch
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *

import torch
from math import sqrt
import numpy as np
import pdb
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


def creatRealDictionary(T, rr, theta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones  = Wones
    for i in range(0,T):
        W1 = torch.mul(torch.pow(rr,i) , torch.cos(i * theta))
        W2 = torch.mul (torch.pow(rr,i) , torch.sin(i *theta) )
        W = torch.cat((Wones,W1,W2),0)
        WVar.append(W.view(1,-1))
    dic = torch.cat((WVar),0)

    return dic

def fista_new(D, Y, lambd,maxIter, gpu_id):
    DtD = torch.matmul(torch.t(D),D)
    L = torch.norm(DtD,2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    x_old = torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id)
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]).cuda(gpu_id) - torch.mul(DtD,linv)
    DtY = torch.mul(DtY,linv)
    Softshrink = nn.Softshrink(lambd)
    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old)
        del y_old
        x_new = Softshrink((Ay + DtY)+1e-6)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        # pdb.set_trace()
        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-5:
            x_old = x_new
            # print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    return x_old

def fista_reweighted(D, Y, lambd, w, maxIter):
    'D: T x 161, Y: N x T x 50, w: N x 161 x 50'
    if len(D.shape) < 3:
        DtD = torch.matmul(torch.t(D), D)
        DtY = torch.matmul(torch.t(D), Y)
    else:
        DtD = torch.matmul(D.permute(0, 2, 1), D)
        DtY = torch.matmul(D.permute(0, 2, 1), Y)
  

    L = torch.norm(DtD, 2)
    Linv = 1/L

    weightedLambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).to(D.device)
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]).to(D.device) - torch.mul(DtD,Linv)
    t_old = 1

    const_xminus = torch.mul(DtY, Linv) - weightedLambd.to(D.device)
    const_xplus = torch.mul(DtY, Linv) + weightedLambd.to(D.device)

    iter = 0

    while iter < maxIter:
        iter +=1
        Ay = torch.matmul(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old-1)/t_new
        y_new = x_new + torch.mul(tt, (x_new-x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old

class DyanEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, lam, gpu_id):
        super(DyanEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        # self.T = T
        self.lam = lam
        self.gpu_id = gpu_id

    def forward(self, x,T):
        'with RH'
        dic = creatRealDictionary(T, self.rr,self.theta, self.gpu_id)

        i = 0
        # ipdb.set_trace()
        w_init = torch.ones(x.shape[0], dic.shape[1], x.shape[2])
        while i < 2:
            temp = fista_reweighted(dic, x, self.lam, w_init, 100)
            'for vector:'
            w = 1 / (torch.abs(temp) + 1e-2)
            w_init = (w/torch.norm(w)) * dic.shape[1]

            'for matrix:'
            # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
            # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
            # pdb.set_trace()

            final = temp
            del temp
            i += 1

        sparseCode = final
        reconst = torch.matmul(dic, sparseCode.cuda(self.gpu_id))
        return sparseCode, dic, reconst
    
    def forward2(self,x, T):
        'no re-weighted'
        dic = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)
        sparseCode = fista_new(dic, x, self.lam, 100, self.gpu_id)
        
        reconst = torch.matmul(dic, sparseCode)
        # return sparseCode, dic, reconst
        return sparseCode, dic, reconst