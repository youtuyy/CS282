import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ConvNet(object):

  """
  [(conv - relu)*2 - 2x2 maxpool]x2 - affine - relu - affine - softmax
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=(32,32,64,64), filter_size=(5,5,5,5),
               hidden_dim=128, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    
    self.params['W1'] = weight_scale * np.random.randn(num_filters[0], C, filter_size[0], filter_size[0])
    self.params['b1'] = np.zeros((1, num_filters[0]))
    self.params['W2'] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1])
    self.params['b2'] = np.zeros((1, num_filters[1]))
    self.params['W3'] = weight_scale * np.random.randn(num_filters[2], num_filters[1], filter_size[2], filter_size[2])
    self.params['b3'] = np.zeros((1, num_filters[2]))
    self.params['W4'] = weight_scale * np.random.randn(num_filters[3], num_filters[2], filter_size[3], filter_size[3])
    self.params['b4'] = np.zeros((1, num_filters[3]))

    self.params['W5'] = weight_scale * np.random.randn(num_filters[3]*H*W/16, hidden_dim)
    self.params['b5'] = np.zeros((1, hidden_dim))
    self.params['W6'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b6'] = np.zeros((1, num_classes))
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    
    conv_param_1 = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
    conv_param_2 = {'stride': 1, 'pad': (W2.shape[2] - 1) / 2}
    conv_param_3 = {'stride': 1, 'pad': (W3.shape[2] - 1) / 2}
    conv_param_4 = {'stride': 1, 'pad': (W4.shape[2] - 1) / 2}
    pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
    dropout_param = {'p': 0.5}
    dropout_param['mode'] = 'test' if y is None else 'train'


    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    a1, cache1 = conv_relu_forward(X, W1, b1, conv_param_1)
    a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param_2, pool_param)
    a3, cache3 = conv_relu_forward(a2, W3, b3, conv_param_3)
    a4, cache4 = conv_relu_pool_forward(a3, W4, b4, conv_param_4, pool_param)
    
    a5, cache5 = affine_relu_forward(a4, W5, b5)
    d5, cache6 = dropout_forward(a5, dropout_param)
    scores, cache7 = affine_forward(d5, W6, b6)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    dd5, dW6, db6 = affine_backward(dscores, cache7)
    da5 = dropout_backward(dd5, cache6)
    da4, dW5, db5 = affine_relu_backward(da5, cache5)
    
    da3, dW4, db4 = conv_relu_pool_backward(da4, cache4)
    da2, dW3, db3 = conv_relu_backward(da3, cache3)
    da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
    dX, dW1, db1 = conv_relu_backward(da1, cache1)
    

    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5
    dW6 += self.reg * W6
    reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5,W6])

    loss = data_loss + reg_loss
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 
             'W5': dW5, 'b5': db5, 'W6': dW6, 'b6': db6}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
