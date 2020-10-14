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

# Some global constants related to the problem
k_1 = torch.empty(1).uniform_(0.05, 0.4)
k_2 = torch.empty(1).uniform_(0.05, 0.4)
k_3 = torch.empty(1).uniform_(0.05, 0.4)
k_4 = torch.empty(1).uniform_(0.05, 0.4)
k_5 = torch.empty(1).uniform_(0.05, 0.4)
k_6 = torch.empty(1).uniform_(0.05, 0.4)
m_1 = torch.empty(1).uniform_(1, 5)
m_2 = torch.empty(1).uniform_(1, 5)
m_3 = torch.empty(1).uniform_(1, 5)
m_4 = torch.empty(1).uniform_(1, 5)
m_5 = torch.empty(1).uniform_(1, 5)
L = 1.5

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--n', default=800, type=int, help='number of points to generate')
    parser.add_argument('--dt', default=0.1, type=float, help='time step')
    parser.add_argument('--std', default=0.0, type=float, help='gaussian noise st dev')
    parser.set_defaults(feature=True)
    return parser.parse_args()

class Tp_ms(nn.Module):
    def forward(self, p):
        global k_1, k_2, k_3, k_4, k_5, k_6, m_1, m_2, m_3, m_4, m_5
        p1 = p[0][0]
        p2 = p[0][1]
        p3 = p[0][2]
        p4 = p[0][3]
        p5 = p[0][4]
        dp1 = p1 / m_1
        dp2 = p2 / m_2
        dp3 = p3 / m_3
        dp4 = p4 / m_4
        dp5 = p5 / m_5
        return torch.tensor([[ dp1, dp2, dp3, dp4, dp5 ]])

class Vq_ms(nn.Module):
    def forward(self, q):
        global k_1, k_2, k_3, k_4, k_5, k_6, m_1, m_2, m_3, m_4, m_5, L
        q1 = q[0][0]
        q2 = q[0][1]
        q3 = q[0][2]
        q4 = q[0][3]
        q5 = q[0][4]
        dq1 = k_1*q1 + k_2*q1 - k_2*q2
        dq2 = k_2*q2 - k_2*q1 + k_3*q2 - k_3*q3
        dq3 = k_3*q3 - k_3*q2 + k_4*q3 - k_4*q4
        dq4 = k_4*q4 - k_4*q3 + k_5*q4 - k_5*q5
        dq5 = k_5*q5 - k_5*q4 + k_6*q5 - k_6*L
        return torch.tensor([[ dq1, dq2, dq3, dq4, dq5 ]])

def generate_data():
    # ~ Check if data exists already ~
    if (os.path.exists("ms.pkl")):
        print('ms.pkl found in local directory')
        return 0

    # ~ Fetch arguments ~
    args = get_args()
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)

    # ~ Instantiate Model ~
    Tp = Tp_ms().to(device) # Kinetic
    Vq = Vq_ms().to(device) # Potential
    model = SSINN(Tp, Vq, fourth_order).to(device)
    dt = args.dt

    # ~ Generate Data ~
    # For this experiment, we generate one trajectory with a single initial state
    t0 = torch.tensor([[0]]).to(device)
    t1 = torch.tensor([[dt]]).to(device)
    p0 = torch.empty(1, 5).uniform_(-0.1, 0.1)
    q0 = torch.tensor([[ 0.25, 0.50, 0.75, 1.00, 1.25 ]])
    points = []
    for i in range(args.n):
        p1, q1 = model(p0, q0, t0, t1)
        ele = (torch.cat((q0, p0), 1), torch.cat((q1, p1), 1))
        if (torch.min(q1) < 0 or torch.max(q1) > L):
            print("WARNING: positions outside of bounds, check constants or rerun")
            return 0
        points.append(ele)
        p0, q0 = p1, q1
        print('{}/{}'.format(len(points),args.n), end='\r')

    # ~ Add Noise if Desired ~
    if (args.std != 0):
        for i in range(len(points)):
            noise1 = torch.empty(1,10).normal_(mean=0,std=args.std)
            noise2 = torch.empty(1,10).normal_(mean=0,std=args.std)
            points[i] = (points[i][0] + noise1, points[i][1] + noise2)

    # ~ Save Model ~
    with open("./ms.pkl", 'wb') as handle:
        pickle.dump(points, handle)
    print('{} Mass-spring state-pairs (dt={}, noise stdev={}) saved in ./ms.pkl'.format(args.n, dt, args.std))

if __name__ == '__main__':
    generate_data()
