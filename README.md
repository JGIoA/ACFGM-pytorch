# acfgm-pytorch

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/acfgm-pytorch?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/acfgm-pytorch)
[![PyPI](https://img.shields.io/pypi/v/ACFGM-pytorch.svg?color=blue)](https://pypi.org/project/ACFGM-pytorch/)
![License](https://img.shields.io/github/license/JGIoA/ACFGM-pytorch.svg?color=blue)

`acfgm-pytorch` is a PyTorch optimizer implementation of the Auto-Conditioned Fast Gradient Method (AC-FGM) from [A simple uniformly optimal method without line search for convex optimization](https://arxiv.org/abs/2310.10082v2) (Version v2).

AC-FGM is an accelerated first-order method designed for convex optimization without the need for estimating the global Lipschitz constant. Instead of asking the user to tune a fixed learning rate, the method estimates local curvature from previous gradients and updates its step size automatically. This package wraps that implementation in `torch.optim.Optimizer` interface so it can be used with tensors, custom objectives, and standard `torch.nn.Module` parameters.

## Features

- Implements the Corollary 2 setup for AC-FGM update described in the paper, with an optional simple line search step through `linesearch=True` for better complexity bound.
- Uses a standard PyTorch closure like `torch.optim.LBFGS`: each optimizer step reevaluates the objective, computes gradients, and returns the loss.
- Includes projection bounds through `lims`, which keeps iterates inside a box constraint such as `[-10, 10]`.
- Works on any device supported by the optimized tensors, including CPU, CUDA and MPS tensors.

## Installation

```bash
pip install acfgm-pytorch
```

## Optimizer Parameters

```python
ACFGM(params, beta=0.1, eps=1e-8, lims=None, linesearch=False)
```

- `params`: iterable of tensors or parameter groups to optimize.
- `beta`: AC-FGM averaging parameter. The implementation accepts values in
  `(0, 1)`; the paper discusses more specific theoretical ranges for particular
  guarantees.
- `eps`: small positive value used to avoid division by zero in curvature and
  norm calculations.
- `lims`: two-element projection interval. If omitted, parameters are projected
  to `[-1, 1]`; pass a wider interval when your problem requires it.
- `linesearch`: choose whether linesearch is used (only in the first iteration) to ensure $\eta_1 \in [\frac{\beta}{4 (1-\beta) L_1}, \frac{1}{3L_1} ]$. The default is `False`.

## Quickstart

## Optimizing Different Parameter Shapes

`ACFGM` can optimize scalar, vector, and higher-dimensional tensor parameters in
the same optimizer instance. This is useful for experiments where the decision
variables are tensors rather than a neural network.

This example places three independent quadratic objectives into one loss:
a scalar target near `2.0`, a vector target near `3.0`, and a rank-3 tensor
target near `4.0`.

```python
import torch

from acfgm import ACFGM

device = "cpu" # or "cuda" "mps" 
scalar_param = torch.tensor(0.5, device=device, requires_grad=True)
vector_param = torch.rand(3, device=device, requires_grad=True)
tensor_param = torch.rand(2, 3, 4, device=device, requires_grad=True)
optimizer = ACFGM(
    [scalar_param, vector_param, tensor_param],
    beta=0.26,
    lims=[-10, 10],
)

for i in range(25):
    def closure():
        optimizer.zero_grad()
        loss = (
            (scalar_param - 2.0).pow(2)
            + (vector_param - 3.0).pow(2).sum()
            + (tensor_param - 4.0).pow(2).sum()
        )
        loss.backward(retain_graph=True)
        return loss
    
    loss = optimizer.step(closure)
    if i % 5 == 0:
        print(f"itr {i}: loss {round(loss.detach().cpu().item(), 5)}")


print(
    scalar_param.detach(),
    vector_param.detach(),
    tensor_param.detach().mean(),
)
```


## Training a Neural Network

Although AC-FGM is designed for smooth convex optimization, the optimizer can also be
applied to standard PyTorch modules. The example below fits a simple
model and checks that the loss decreases.


```python
import torch
from torch import nn

from acfgm import ACFGM

device = "cpu" # or "cuda" "mps" 
torch.manual_seed(0)
train_x = torch.linspace(-1.0, 1.0, 16, device=device).unsqueeze(1)
train_y = 2.0 * train_x - 1.0

model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)).to(device)
optimizer = ACFGM(model.parameters(), beta=0.26, lims=[-10, 10])
loss_fn = nn.MSELoss()

for i in range(25):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    if i % 5 == 0:
        print(f"itr {i}: loss {round(loss.detach().cpu().item(), 5)}")
```


## Reference

```bibtex
@misc{li2024simpleuniformlyoptimalmethod,
  title = {A simple uniformly optimal method without line search for convex optimization},
  author = {Tianjiao Li and Guanghui Lan},
  year = {2024},
  eprint = {2310.10082},
  archivePrefix = {arXiv},
  primaryClass = {math.OC},
  url = {https://arxiv.org/abs/2310.10082v2}
}
```
