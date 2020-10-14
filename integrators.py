###############################################################################
# General Information
###############################################################################
# Sparse Symplectically Integrated Neural Networks (2020)
# Paper: https://arxiv.org/abs/2006.12972
# Daniel DiPietro, Shiying Xiong, Bo Zhu

import torch
import numpy as np

# Fourth-order symplectic integrator
def fourth_order(p0, q0, t0, t1, Tp, Vq, eps=0.1):
    n_steps = np.round((torch.abs(t1 - t0)/(eps * 4)).max().item())
    h = (t1 - t0)/n_steps
    kp = p0
    kq = q0
    c = torch.Tensor([0.5/(2.-2.**(1./3.)),
         (0.5-2.**(-2./3.))/(2.-2.**(1./3.)),
         (0.5-2.**(-2./3.))/(2.-2.**(1./3.)),
         0.5/(2.-2.**(1./3.))])
    d = torch.Tensor([1./(2.-2.**(1./3.)),
         -2.**(1./3.)/(2.-2.**(1./3.)),
         1./(2.-2.**(1./3.)),0.])
    for i_step in range(int(n_steps)):
        for j in range(4):
            tp = kp
            tq = kq + c[j] * Tp(kp) * h
            kp = tp - d[j] * Vq(tq) * h
            kq = tq
    return kp, kq

# RK4 integrator
def rk4(p0, q0, t0, t1, Tp, Vq, eps=0.1):
    n_steps = np.round((torch.abs(t1 - t0)/(eps * 4)).max().item())
    h = (t1 - t0)/n_steps
    kp = p0
    kq = q0
    for i_step in range(int(n_steps)):
        p1 = -Vq(kq)
        q1 = Tp(kp)
        p2 = -Vq(kq+0.5*q1*h)
        q2 = Tp(kp+0.5*p1*h)
        p3 = -Vq(kq+0.5*q2*h)
        q3 = Tp(kp+0.5*p2*h)
        p4 = -Vq(kq+q3)
        q4 = Tp(kp+p3)
        kp = kp+(p1+2*p2+2*p3+p4)*h/6
        kq = kq+(q1+2*q2+2*q3+q4)*h/6
    return kp, kq
