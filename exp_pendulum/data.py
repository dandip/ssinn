###############################################################################
# General Information
###############################################################################
# Sparse Symplectically Integrated Neural Networks (2020)
# Paper: https://arxiv.org/abs/2006.12972
# Daniel DiPietro, Shiying Xiong, Bo Zhu

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional
import numpy as np
import math
import random
import logging
import pickle
import argparse

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from integrators import fourth_order, rk4
from function_spaces import gradient_wrapper, bivariate_poly
from ssinn import SSINN
from utils import get_logger

l = 1
m = 2
g = 1

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--n', default=200, type=int, help='number of points to generate')
    parser.add_argument('--dt', default=0.1, type=float, help='time step')
    parser.add_argument('--std', default=0.0, type=float, help='gaussian noise st dev')
    parser.set_defaults(feature=True)
    return parser.parse_args()

class Tp_pend(nn.Module):
    def forward(self, p):
        global l, m, g
        p1 = p[0][0]

        dp1 = p1 / (m*(l**2))

        return torch.tensor([[ dp1 ]])
class Vq_pend(nn.Module):
    def forward(self, q):
        global l, m, g
        q1 = q[0][0]

        dq1 = m*g*l*torch.sin(q1)

        return torch.tensor([[ dq1 ]])

def generate_data():
    # ~ Check if data exists already ~
    if (os.path.exists("pend.pkl")):
        print('pend.pkl found in local directory')
        return 0

    # ~ Fetch arguments ~
    args = get_args()
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)

    # ~ Instantiate Model ~
    Tp = Tp_pend().to(device) # Kinetic
    Vq = Vq_pend().to(device) # Potential
    model = SSINN(Tp, Vq, fourth_order).to(device)
    dt = args.dt

    # ~ Generate Data ~
    # For this experiment, we generate one trajectory with a single initial state
    q0 = torch.tensor([[ 1.4 ]])
    p0 = torch.tensor([[ 0. ]])
    points = []
    for i in range(args.n):
        t0 = torch.tensor([[0.]]).to(device)
        t1 = torch.tensor([[dt]]).to(device)
        p1, q1 = model(p0, q0, t0, t1)
        ele = (torch.cat((q0, p0), 1), torch.cat((q1, p1), 1))
        points.append(ele)
        p0, q0 = p1, q1
        print('{}/{}'.format(len(points),args.n), end='\r')

    # ~ Add Noise if Desired ~
    if (args.std != 0):
        for i in range(len(points)):
            noise1 = torch.empty(1,4).normal_(mean=0,std=args.std)
            noise2 = torch.empty(1,4).normal_(mean=0,std=args.std)
            points[i] = (points[i][0] + noise1, points[i][1] + noise2)

    # ~ Save Model ~
    with open("./pend.pkl", 'wb') as handle:
        pickle.dump(points, handle)
    print('{} Pendulum state-pairs (dt={}, noise stdev={}) saved in ./pend.pkl'.format(args.n, dt, args.std))

if __name__ == '__main__':
    generate_data()
