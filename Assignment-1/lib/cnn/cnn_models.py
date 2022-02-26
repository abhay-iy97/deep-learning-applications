from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3,3,3,1,0,0.02,'conv'),
            MaxPoolingLayer(2,2,'max_pool'),
            flatten('flatten'),
            fc(27, 5, 0.02, name="fc_test")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3,5,16,1,0,5e-2,'conv'),
            leaky_relu(name="relu1"),
            MaxPoolingLayer(4,2,'max_pool'),
            flatten('flatten'),
            fc(2704, 512, 5e-2, name="fc1"),
            leaky_relu(name="relu2"),
            fc(512, 128, 5e-2, name="fc2"),
            leaky_relu(name="relu3"),
            fc(128, 10, 5e-2, name="fc3")
            
            ########### END ###########
        )
