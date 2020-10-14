# Sparse Symplectically Integrated Neural Networks

Daniel DiPietro, Shiying Xiong, Bo Zhu (2020)

*Paper*: https://arxiv.org/abs/2006.12972

## Summary

Sparse Symplectically Integrated Neural Networks (SSINNs) are a novel model for learning Hamiltonian dynamical systems from data. SSINNs combine fourth-order symplectic integration with a learned parameterization of the Hamiltonian obtained using sparse regression through a mathematically elegant function space. This allows for interpretable models that incorporate symplectic inductive biases and have low memory requirements. SSINNs often successfully converge to true governing equations from highly limited and noisy data.

## Usage

SSINNs may be applied to a variety of tasks using the code in `function_spaces.py`, `integrators.py`, and `ssinn.py`.

## Assembling an SSINN

SSINNs can be readily applied to novel dynamical systems.

## Dependencies
* PyTorch
* NumPy
