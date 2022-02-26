from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        w = self.params[self.w_name]
        output_height = (input_size[1] + 2*self.padding - self.kernel_size)//self.stride + 1
        output_width = (input_size[2]  + 2*self.padding - self.kernel_size)//self.stride + 1
        batch_size = input_size[0]
        out_channels = w.shape[3]
        output_shape = [batch_size, output_height, output_width, out_channels]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        #pad the input image according to self.padding (see np.pad)
        padded_img = np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)
        w = self.params[self.w_name]
        b = self.params[self.b_name]
        output = np.zeros((img.shape[0],output_height,output_width,w.shape[3]))

        #iterate over output dimensions, moving by self.stride to create the output

        for height_index in range(output_height):
            start_h = height_index * self.stride
            end_h = start_h + self.kernel_size
            for width_index in range(output_width):

                    start_w = width_index * self.stride
                    end_w = start_w + self.kernel_size

                    image_portion = padded_img[:,start_h:end_h, start_w:end_w,:,np.newaxis]
                    kernel_matrix = w[np.newaxis,:,:,:,:]
                    output[:,height_index,width_index,:] = np.sum(np.multiply(image_portion, kernel_matrix), axis=(1,2,3))
        output += b
        self.meta = img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        number_of_examples, input_height, input_width, input_channels = img.shape
        w = self.params[self.w_name]
        _, height, width, channels = dprev.shape
        
        dimg = np.zeros((number_of_examples, input_height, input_width, input_channels))
        dW = np.zeros((self.kernel_size, self.kernel_size, input_channels, channels))
        db = np.zeros(channels)
        img_padded = np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)
        dimg_padded = np.pad(dimg, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)
        
        for height_index in range(height):
            start_h = height_index * self.stride
            end_h = start_h + self.kernel_size

            for width_index in range(width):
                start_w = width_index * self.stride
                end_w = start_w + self.kernel_size

                dimg_padded[:,start_h:end_h, start_w:end_w, :] += np.sum(w[np.newaxis,:,:,:,:] * dprev[:,height_index:height_index+1,width_index:width_index+1,np.newaxis,:], axis = 4)

                dW[:,:,:,:] += np.sum(img_padded[:,start_h:end_h, start_w:end_w, :, np.newaxis] * dprev[:,height_index:height_index+1,width_index:width_index+1,np.newaxis,:], axis=0)

                db[:] += np.sum(dprev[:,height_index,width_index,:], axis=0)


        if self.padding != 0:
            dimg[:,:,:,:] = dimg_padded[:,self.padding:-self.padding, self.padding:-self.padding,:]
        else:
            dimg[:,:,:,:] = dimg_padded[:,:,:,:]
        self.grads[self.w_name] = dW
        self.grads[self.b_name] = db
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        
        number_of_examples, input_height, input_width, input_channels = img.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        output = np.zeros((number_of_examples,output_height,output_width,input_channels))
        #iterate over output dimensions, moving by self.stride to create the output

        for height_index in range(output_height):
            start_h = height_index * self.stride
            end_h = start_h + self.pool_size
            for width_index in range(output_width):
                start_w = width_index * self.stride
                end_w = start_w + self.pool_size

                image_portion = img[:,start_h:end_h, start_w:end_w,:]
                output[:,height_index,width_index,:] = np.amax(image_portion, axis=(1,2))
        self.meta = img
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        number_of_examples, input_height, input_width, input_channels = img.shape
        for height_index in range(h_out):
            start_h = height_index * self.stride
            end_h = start_h + h_pool        
            for width_index in range(w_out):
                    start_w = width_index * self.stride
                    end_w = start_w + w_pool

                    image_portion = img[:, start_h:end_h, start_w:end_w, :]
                    value = np.where(image_portion == np.amax(image_portion, axis=(1,2), keepdims=True), 1, 0)
                    dimg[:,start_h:end_h,start_w:end_w,:] += np.multiply(value, dprev[:, height_index:height_index+1, width_index:width_index+1, :])

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg

