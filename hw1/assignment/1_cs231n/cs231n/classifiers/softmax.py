import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.

  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_x = X.shape[1]
  # loss
  z = np.matmul(W, X) # CxN
  exp_z = np.exp(z+np.exp(-10))
  exp_prob = exp_z / np.sum(exp_z, axis=0, keepdims=True)
  loss = np.sum(-np.log(exp_prob[y, range(num_x)]))
  loss /= num_x
  loss += 0.5 * reg * np.sum(W**2)

  # gradient
  dz = exp_z / np.sum(exp_z, axis=0, keepdims=True)
  dz[y, range(num_x)] -= 1
  dW = np.dot(dz, X.T)
  dW /= num_x
  dW += reg * W
  

  

        
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
