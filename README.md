# Sparse Symplectically Integrated Neural Networks

Daniel DiPietro, Shiying Xiong, Bo Zhu (2020)

*Paper*: https://arxiv.org/abs/2006.12972

## Summary

Sparse Symplectically Integrated Neural Networks (SSINNs) are a novel model for learning Hamiltonian dynamical systems from data. SSINNs combine fourth-order symplectic integration with a learned parameterization of the Hamiltonian obtained using sparse regression through a mathematically elegant function space. This allows for interpretable models that incorporate symplectic inductive biases and have low memory requirements. SSINNs often successfully converge to true governing equations from highly limited and noisy data.

<p align="center">
    <img src="LearningAnimation.gif" alt="henon-heiles trajectory sample" width="500"/>
</p>

## Installation

Clone this repo:
```
git clone https://github.com/dandip/ssinn.git
```

## Data and Training

Each folder contains a corresponding `data.py`, which generates the necessary dataset in the working directory, and `train.py`, which trains the SSINN on the given dataset. If the dataset is not in the directory when `train.py` is executed, it will call `data.py` and generate the dataset using default arguments. To generate datasets with noise, call `data.py` with the desired arguments prior to calling `train.py`. Similarly, each `train.py` file may be passed arguments to customize training.

To train a SSINN on the following experiments:
* Henon-Heiles: `python3 exp_henon_heiles/train.py`
* Coupled-Oscillator System: `python3 exp_coupled_oscillator/train.py`
* Five Particle Mass-Spring: `python3 exp_mass_spring/train.py`
* Pendulum System: `python3 exp_pendulum/train.py`

## Dependencies
* PyTorch
* NumPy

## Citation

If you find this code or technique useful, please consider citing:
```
@incollection{ssinn2020,
title = {Sparse Symplectically Integrated Neural Networks},
author = {DiPietro, Daniel M. and Xiong, Shiying and Zhu, Bo},
booktitle = {Advances in Neural Information Processing Systems 34},
year = {2020}
```
