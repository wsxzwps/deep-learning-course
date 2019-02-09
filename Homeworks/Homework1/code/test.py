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
    batch_size = x.shape[0]
    out = np.dot(x, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache

def torch_implementation(x,w,b):
    layer = nn.Linear(w.shape[0], w.shape[1])
    with torch.no_grad():
        layer.weight = nn.Parameter(w)
        layer.bias = nn.Parameter(b)
        out = layer.forward(x)
    return out

w = np.random.rand(9, 8)
b = np.random.rand(8)
x = np.random.rand(10, 9)
out1,_ = fc_forward(x, w, b)

weights = torch.Tensor(w.T).double()
bias = torch.Tensor(b).double()
data = torch.tensor(x)
out2 = torch_implementation(data, weights, bias)
print(out1)
print(out2)