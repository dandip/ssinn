###############################################################################
# General Information
###############################################################################
# Sparse Symplectically Integrated Neural Networks (2020)
# Paper: https://arxiv.org/abs/2006.12972
# Daniel DiPietro, Shiying Xiong, Bo Zhu

###############################################################################
# Dependencies
###############################################################################

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional
import numpy as np
import math
import random
from datetime import datetime
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
import data

###############################################################################
# Argument parsing
###############################################################################

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--degree', default=3, type=int, help='bivariate polynomial degree')
    parser.add_argument('--dt', default=0.1, type=float, help='time-step in training data')
    parser.add_argument('--shuffle_data', default=False, type=bool, help='shuffle data')
    parser.add_argument('--training_split', default=0.8, type=float, help='training/test split')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--decay', default=True, type=bool, help='learning rate decay (on/off)')
    parser.add_argument('--n_epochs', default=6, type=int, help='epochs')
    parser.add_argument('--regularization', default=1e-3, type=float, help='l1 regularization')
    parser.add_argument('--tol', default=1e-2, type=float, help='integrator tolerance')
    parser.add_argument('--freq', default=250, type=int, help='message frequency')
    parser.add_argument('--verbose', default=True, type=bool, help='verbose')
    parser.add_argument('--use_gpu', default=False, type=bool, help='use GPU')
    parser.add_argument('--gpu', default=0, type=int, help='GPU number')
    parser.set_defaults(feature=True)
    return parser.parse_args()

###############################################################################
# Training loop
###############################################################################

def main():
    # ~ Fetch arguments and logger ~
    args = get_args()
    logger = get_logger(logpath=os.path.join('./hh.log'), filepath=os.path.abspath(__file__))

    # ~ Set GPU/CPU if desired ~
    if (args.use_gpu and torch.cuda.is_available()):
        device = torch.device('cuda:' + str(args.gpu))
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type(torch.FloatTensor)

    # ~ Fetch and Split Data ~
    data.generate_data()
    dataset = pickle.load(open("./hh.pkl", "rb" ))
    if (args.shuffle_data):
        random.shuffle(dataset)
    train = dataset[:int(args.training_split*len(dataset))]
    test = dataset[int(args.training_split*len(dataset)):]

    # ~ Instantiate Model ~
    Tp = bivariate_poly(degree=args.degree).to(device) # Kinetic
    Vq = bivariate_poly(degree=args.degree).to(device) # Potential
    _Tp = gradient_wrapper(Tp)
    _Vq = gradient_wrapper(Vq)
    model = SSINN(_Tp, _Vq, fourth_order, tol=args.tol).to(device)

    # ~ Train Model ~
    train_model(model, train, test, args, device, logger)

    # ~ Save Model ~
    torch.save({'state_dict': model.state_dict()}, './hh_model.pth')

def train_model(model, train, test, args, device, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    p_criterion = nn.L1Loss().to(device)
    q_criterion = nn.L1Loss().to(device)
    lr_decay = {2: 0.1, 3: 0.01, 4: 0.001} # Customize learning rate decay here
    dt = args.dt

    # Iterate over each epoch
    for i in range(args.n_epochs):
        log(args.verbose, logger, "epoch (train) " + str(i+1) + " | " + datetime.now().strftime("%H:%M:%S"))

        # Decay learning rate if desired
        if (i in lr_decay and args.decay):
            logger.info("learning rate decayed to " + str(args.lr*lr_decay[i]))
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr*lr_decay[i]

        # Iterate over all training points
        counter = 0
        for point in train:
            optimizer.zero_grad()

            p0, q0, t0, t1, p1T, q1T = format_data(point, dt, device)
            p1N, q1N = model(p0, q0, t0, t1)

            # Accumulate losses and optimize
            l1_regularization = 0.
            for param in model.parameters():
                l1_regularization += param.abs().sum()
            loss = p_criterion(p1N, p1T) + q_criterion(q1N, q1T) + args.regularization*l1_regularization
            if (not torch.isnan(loss).item()):
                loss.backward()
                optimizer.step()
                scheduler.step()
                counter += 1

            # Print message if desired
            if (counter % args.freq == 0 and args.verbose):
                with torch.no_grad():
                    logger.info("\tTp:" + str(model.Tp.space.fc1.weight[0]))
                    logger.info("\tVq:" + str(model.Vq.space.fc1.weight[0]))
                logger.info("\t" + str(counter) + " training points iterated | Loss: " + str(float(loss.item())))

        # Iterate over all testing points
        log(args.verbose, logger, "epoch (test) " + str(i+1) + " | " + datetime.now().strftime("%H:%M:%S"))
        distance_euclideans, momentum_euclideans, losses = [], [], []
        for point in test:
            optimizer.zero_grad()

            p0, q0, t0, t1, p1T, q1T = format_data(point, dt, device)
            p1N, q1N = model(p0, q0, t0, t1)

            # Accumulate losses, but do not backprop
            l1_regularization = 0.
            for param in model.parameters():
                l1_regularization += param.abs().sum()
            loss = p_criterion(p1N, p1T) + q_criterion(q1N, q1T) + args.regularization*l1_regularization

            if (not torch.isnan(loss).item()):
                distance_euclidean, momentum_euclidean = euclideans(p1T, p1N, q1T, q1N)
                distance_euclideans.append(distance_euclidean)
                momentum_euclideans.append(momentum_euclidean)
                losses.append(float(loss.item()))

        # Print test metrics
        log(args.verbose, logger, "\tAverage position euclidean-norm: {}".format(sum(distance_euclideans)/len(distance_euclideans)))
        log(args.verbose, logger, "\tAverage momentum euclidean-norm: {}".format(sum(momentum_euclideans)/len(momentum_euclideans)))
        log(args.verbose, logger, "\tAverage loss: {}".format(sum(losses)/len(losses)))
        log(args.verbose, logger, "")

def format_data(point, dt, device):
    t0 = torch.tensor([[0.]]).to(device)
    t1 = torch.tensor([[dt]]).to(device)
    p0 = Variable(point[0][:, 2:4].to(device), requires_grad=True) # [[vx0, vy0]]
    q0 = Variable(point[0][:, 0:2].to(device), requires_grad=True) # [[x0, y0]]
    p1T = point[1][:, 2:4].to(device) # [[vx1, vy1]]
    q1T = point[1][:, 0:2].to(device) # [[x1, y1]]
    return p0, q0, t0, t1, p1T, q1T

def euclideans(p1T, p1N, q1T, q1N):
    with torch.no_grad():
        true_vx, true_vy, true_x, true_y = p1T[0][0], p1T[0][1], q1T[0][0], q1T[0][1]
        pred_vx, pred_vy, pred_x, pred_y = p1N[0][0], p1N[0][1], q1N[0][0], q1N[0][1]
        distance_euclidean = math.sqrt((pred_x-true_x)**2 + (pred_y-true_y)**2)
        momentum_euclidean = math.sqrt((pred_vx-true_vx)**2 + (pred_vy-true_vy)**2)
        return distance_euclidean, momentum_euclidean

def log(verbose, logger, message):
    if (verbose):
        logger.info(message)

if __name__ == '__main__':
    main()
