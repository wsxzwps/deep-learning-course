from builtins import range
import numpy as np

import torch
import torch.nn as nn

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    out = np.dot(x, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache

def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.dot(dout, w.T)
    dw = np.dot(x.T, dout)
    db = np.average(dout, axis = 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def torch_implementation(x,w,b):
    layer = nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        layer.weight = nn.Parameter(w)
        layer.bias = nn.Parameter(b)
        out = layer.forward(x)
        layer.b
    return out

w = np.random.rand(9, 8)
b = np.random.rand(8)
x = np.random.rand(10, 9)
dout = np.random.rand(10,8)
out1,cache = fc_forward(x, w, b)

dx, dw, db = fc_backward(dout, cache)
print(dx.shape, dw.shape, db.shape)

weights = torch.Tensor(w.T).double()
bias = torch.Tensor(b).double()
data = torch.tensor(x)
out2 = torch_implementation(data, weights, bias)
print(out1)
print(out1.shape)
print(out2)
print(out2.shape)