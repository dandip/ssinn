###############################################################################
# General Information
###############################################################################
# Sparse Symplectically Integrated Neural Networks (2020)
# Paper: https://arxiv.org/abs/2006.12972
# Daniel DiPietro, Shiying Xiong, Bo Zhu

import torch
from torch import nn

class SSINN(nn.Module):
    def __init__(self, Tp, Vq, solver, tol=1e-3):
        super(SSINN, self).__init__()
        self.Tp = Tp
        self.Vq = Vq
        self.tol = tol
        self.solver = solver

    def forward(self, p0, q0, t0, t1):
        p, q = self.solver(p0, q0, t0, t1, self.Tp, self.Vq, self.tol)
        return p, q
