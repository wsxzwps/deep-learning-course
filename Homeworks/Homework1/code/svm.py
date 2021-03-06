import numpy as np

from layers import *

class SVM(object):
    """
    A binary SVM classifier with optional hidden layers.
    
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
  
    def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the model. Weights            #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases (if any) using the keys 'W2' and 'b2'.                        #
        ############################################################################
        if hidden_dim:
            w1 = np.random.normal(scale=weight_scale,size=(input_dim,hidden_dim))
            b1 = np.zeros(hidden_dim)
            self.params['W1'] = w1
            self.params['b1'] = b1

            w2 = np.random.normal(scale=weight_scale,size=(hidden_dim,1))
            b2 = np.zeros(1)
            self.params['W2'] = w2
            self.params['b2'] = b2
        else:
            w1 = np.random.normal(scale=weight_scale,size=(input_dim,1))
            b1 = np.zeros(1)
            self.params['W1'] = w1
            self.params['b1'] = b1    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, D)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N,) where scores[i] represents the classification 
        score for X[i].

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """  
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the model, computing the            #
        # scores for X and storing them in the scores variable.                    #
        ############################################################################
        fc1, cache1 = fc_forward(X,self.params['W1'],self.params['b1'])
        fc2, cache2 = None, None
        if 'W2' in self.params:
            act1, cache_act1 = relu_forward(fc1)
            scores, cache2 = fc_forward(act1, self.params['W2'], self.params['b2'])
        else:
            scores = fc1
        
        scores = np.squeeze(scores, axis=1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the model. Store the loss          #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss and make sure that grads[k] holds the gradients for self.params[k]. #
        # Don't forget to add L2 regularization.                                   #
        #                                                                          #
        ############################################################################
        for i in range(y.shape[0]):
            if y[i]==0:
                y[i] = -1
        loss, dloss = svm_loss(scores, y)
        dloss = np.expand_dims(dloss, axis=1)
        l2w1 = np.sum(np.square(self.params['W1']))
        loss += 0.5 * self.reg * l2w1
        if 'W2' in self.params:
            l2w2 = np.sum(np.square(self.params['W2']))
            loss += 0.5 * self.reg * l2w2
            dx2, dw2, db2 = fc_backward(dloss, cache2)
            dw2 += self.reg * self.params['W2']
            grads['W2'] = dw2
            grads['b2'] = db2

            dact1 = relu_backward(dx2, cache_act1)

            dx1, dw1, db1 = fc_backward(dact1, cache1)
            dw1 += self.reg * self.params['W1']
            grads['W1'] = dw1
            grads['b1'] = db1
        else:
            dx1, dw1, db1 = fc_backward(dloss, cache1)
            dw1 += self.reg * self.params['W1']
            grads['W1'] = dw1
            grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
