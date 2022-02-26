from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)
        
        #### MLP ####
        if not hasattr(self.net, "preprocess") and \
           not hasattr(self.net, "rnn") and \
           not hasattr(self.net, "postprocess"):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def update(self, layer):
        for n, dv in layer.grads.items():
            layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def update(self, layer):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        for param, grad in zip(layer.params, layer.grads):
            for layer_index in range(len(layer.grads)):
                if param not in self.velocity.keys():
                    self.velocity[param] = np.zeros(layer.params[param].shape)
                self.velocity[param] = self.momentum * self.velocity[param] - self.lr * layer.grads[grad]
                layer.params[param] += self.velocity[param]             
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def update(self, layer):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for param, grad in zip(layer.params, layer.grads):
            for layer_index in range(len(layer.grads)):
                if param not in self.cache.keys():
                    self.cache[param] = np.zeros(layer.params[param].shape)
                self.cache[param] = self.decay * self.cache[param] + (1 - self.decay) * layer.grads[grad] ** 2
                layer.params[param] -= (self.lr * layer.grads[grad])/(np.sqrt(self.cache[param] + self.eps))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        for param, grad in zip(layer.params, layer.grads):
            for layer_index in range(len(layer.grads)):
                self.t = self.t + 1
                if param not in self.mt.keys():
                    self.mt[param] = np.zeros(layer.params[param].shape)
                self.mt[param] = self.beta1 * self.mt[param] + (1 - self.beta1) * layer.grads[grad]
                if param not in self.vt.keys():
                    self.vt[param] = np.zeros(layer.params[param].shape)
                self.vt[param] = self.beta2 * self.vt[param] + (1 - self.beta2) * layer.grads[grad] ** 2
                mt_hat = self.mt[param] / (1 - np.power(self.beta1, self.t))
                vt_hat = self.vt[param] / (1 - np.power(self.beta2, self.t))
                layer.params[param] -= self.lr * mt_hat / (np.sqrt(vt_hat) + self.eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
