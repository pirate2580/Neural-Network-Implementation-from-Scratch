"""
A module for the common loss functions for a neural network
Author: Naoroj Farhan
Date: Thursday, May 23, 2024
"""

import numpy as np

class Loss:
  """
  Abstract representation of loss function used to evaluate the performance of a neural network model
  """

  def evaluate(self, Y_pred: np.ndarray, Y_target: np.ndarray) -> np.ndarray:
    """
    Return the loss for a neural network as a (1, 1) np array
    """
    raise NotImplementedError
  def derivative(self, Y_pred: np.ndarray, Y_target: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the loss function with respect to activation values
    of the final layer ie dJ/dA[l]. This will be returned as a vector of shape (nl, m)
    """
    raise NotImplementedError

class MeanSquaredError(Loss):
  """
  Implementation of mean squared error for a set of m training examples where m is the number of training examples
  which is equal to the number of columns
  """
  def evaluate(self, Y_pred: np.ndarray, Y_target: np.ndarray) -> np.ndarray:
    """
    Vectorized implementation of Mean squared error loss function averaged over all m training examples.
    Mean squared error loss is deal for regression problems. This implementation assumes the number of columns represents the number of training examples
    n features by m training examples is Y_pred -> 1x1
    Function J(A[l]) = average for each of m examples: (Y-Y*)^2

    Test(s):
    >>> Y_pred = np.array([[1, 2], [1, 3]])     # this would be an example of a neural network with 2 features and 2 training examples where 2 features are along columns and 2 training exmaples are along rows
    >>> Y_target = np.array([[3, 3], [3, 3]])
    >>> loss = MeanSquaredError()
    >>> expected_output = 4.5
    >>> np.testing.assert_allclose(loss.evaluate(Y_pred, Y_target), expected_output, rtol=1e-5) is None
    True
    """
    m = Y_pred.shape[1]
    return np.sum(np.sum((Y_pred - Y_target) ** 2, axis = 0)) / m
  
  def derivative(self, Y_pred: np.ndarray, Y_target: np.ndarray) -> np.ndarray:
    """
    Vectorized implementation to find the derivative of the Mean Squared Error loss function with respect to the activation of last layer
    Y_pred is expected to be of shape: (nL, m) where nL is # of features of last layer and m columns for each of m training example
    
    This function returns dJ/dA[l], a (nL, m) matrix representing the gradient for each training example
    Test(s):
    >>> Y_pred = np.array([[1, 1, 1], [1, 1, 1]])     # this would be an example of a neural network with 2 features and 3 training examples
    >>> Y_target = np.array([[3, 3, 3], [3, 3, 3]])
    >>> loss = MeanSquaredError()
    >>> expected_output = np.array([[-4, -4, -4], [-4, -4, -4]]) # nL x m matrix
    >>> np.testing.assert_allclose(loss.derivative(Y_pred, Y_target), expected_output, rtol=1e-5) is None
    True
    """
    m = Y_pred.shape[1]
    return 2 * (Y_pred - Y_target)


if __name__ == '__main__':
  import doctest
  doctest.testmod()