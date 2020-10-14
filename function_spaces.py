###############################################################################
# General Information
###############################################################################
# Sparse Symplectically Integrated Neural Networks (2020)
# Paper: https://arxiv.org/abs/2006.12972
# Daniel DiPietro, Shiying Xiong, Bo Zhu

import torch
from torch import nn

# Used to wrap each function space so that the forward pass obtains the gradient
class gradient_wrapper(nn.Module):
    def __init__(self, space):
        super(gradient_wrapper, self).__init__()
        self.space = space

    def forward(self, x):
        return torch.autograd.grad(self.space(x), x, create_graph=True)[0]

###############################################################################
# Used for Henon-Heiles/Coupled Oscillator
###############################################################################

# Bivariate polynomial function space. For use with a single point in 2D space.
class bivariate_poly(nn.Module):
    def __init__(self, degree=3):
        super(bivariate_poly, self).__init__()
        self.degree = degree
        self.fc1 = nn.Linear(degree**2 + 2*degree, 1, bias=False)

    def forward(self, x):
        xc = torch.ones([self.degree])*(x[0][0])
        yc = torch.ones([self.degree])*(x[0][1])
        xc_pow = torch.pow(xc, torch.Tensor([i for i in range(1, self.degree+1)]))
        yc_pow = torch.pow(yc, torch.Tensor([i for i in range(1, self.degree+1)]))
        combos = torch.ger(xc_pow, yc_pow).flatten()
        input = torch.cat((xc_pow, yc_pow, combos))
        out = self.fc1(input)
        return out

###############################################################################
# Used for Mass-Spring
###############################################################################

# Polynomial space with 5 input variables. Only supports degree 2, 3, 4, 6, or 10.
class fivevariate_poly(nn.Module):
    def __init__(self, degree):
        super(fivevariate_poly, self).__init__()
        self.degree = degree
        if (degree == 2):
            self.fc1 = nn.Linear(50, 1, bias=False)
        elif (degree == 3):
            self.fc1 = nn.Linear(105, 1, bias=False)
        elif (degree == 4):
            self.fc1 = nn.Linear(180, 1, bias=False)
        elif (degree == 6):
            self.fc1 = nn.Linear(390, 1, bias=False)
        elif (degree == 10):
            self.fc1 = nn.Linear(1050, 1, bias=False)

    def forward(self, x):
        x1 = torch.ones([self.degree])*(x[0][0])
        x2 = torch.ones([self.degree])*(x[0][1])
        x3 = torch.ones([self.degree])*(x[0][2])
        x4 = torch.ones([self.degree])*(x[0][3])
        x5 = torch.ones([self.degree])*(x[0][4])
        x1_pow = torch.pow(x1, torch.Tensor([i for i in range(1, self.degree+1)]))
        x2_pow = torch.pow(x2, torch.Tensor([i for i in range(1, self.degree+1)]))
        x3_pow = torch.pow(x3, torch.Tensor([i for i in range(1, self.degree+1)]))
        x4_pow = torch.pow(x4, torch.Tensor([i for i in range(1, self.degree+1)]))
        x5_pow = torch.pow(x5, torch.Tensor([i for i in range(1, self.degree+1)]))
        x1x2 = torch.ger(x1_pow, x2_pow).flatten()
        x1x3 = torch.ger(x1_pow, x3_pow).flatten()
        x1x4 = torch.ger(x1_pow, x4_pow).flatten()
        x1x5 = torch.ger(x1_pow, x5_pow).flatten()
        x2x3 = torch.ger(x2_pow, x3_pow).flatten()
        x2x4 = torch.ger(x2_pow, x4_pow).flatten()
        x2x5 = torch.ger(x2_pow, x5_pow).flatten()
        x3x4 = torch.ger(x3_pow, x4_pow).flatten()
        x3x5 = torch.ger(x3_pow, x5_pow).flatten()
        x4x5 = torch.ger(x4_pow, x5_pow).flatten()
        input = torch.cat((x1_pow, x2_pow, x3_pow, x4_pow, x5_pow,
                           x1x2, x1x3, x1x4, x1x5,
                           x2x3, x2x4, x2x5,
                           x3x4, x3x5,
                           x4x5))
        out = self.fc1(input)
        return out

###############################################################################
# Used for Pendulum
###############################################################################

# Univariate trinomial. For use with a single point in 1D space (angular data).
class univariate_trinomial(nn.Module):
    def __init__(self):
        super(univariate_trinomial, self).__init__()
        self.fc1 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        x_power = torch.cat(3*[x]).squeeze(1)
        x_power = torch.pow(x_power, torch.Tensor([i for i in range(1, 4)]))
        out = self.fc1(x_power)
        return out

class univariate_fivenomial(nn.Module):
    def __init__(self):
        super(univariate_fivenomial, self).__init__()
        self.fc1 = nn.Linear(5, 1, bias=False)

    def forward(self, x):
        x_power = torch.cat(5*[x]).squeeze(1)
        x_power = torch.pow(x_power, torch.Tensor([i for i in range(1, 6)]))
        out = self.fc1(x_power)
        return out

# Univariate trigonometric space. For use with a single point in 1D space (angular data).
class univariate_trigonometric_term(nn.Module):
    def __init__(self):
        super(univariate_trigonometric_term, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.trig = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        x_sin = torch.sin(self.trig(x.squeeze(1)) + x.squeeze(1))
        out = self.fc1(x_sin)
        return out

# Univariate trigonometric combined with third degree polynomial space.
# For use with a single point in 1D space (angular data).
class univariate_trig_tri_mixed(nn.Module):
    def __init__(self):
        super(univariate_trig_tri_mixed, self).__init__()
        self.fc1 = nn.Linear(4, 1, bias=False)
        self.trig = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        x_power = torch.cat(3*[x]).squeeze(1)
        x_power = torch.pow(x_power, torch.Tensor([i for i in range(1, 4)]))
        x_sin = torch.sin(self.trig(x.squeeze(1)) + x.squeeze(1))
        input = torch.cat((x_power, x_sin))
        out = self.fc1(input)
        return out
