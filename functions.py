"""
A module for common non-linearity functions in a neural network and their derivatives

Author: Naoroj Farhan
Date: Thursday, May 23, 2024
"""

import numpy as np

class Function:
  """
  Abstract representation for non-linearity function applied to a vector in a neural network after 
  weights and biases have been applied
  """

  def activation(self, Z: np.ndarray) -> np.ndarray:
    """
    Given Z[l] = W[l] * A[l-1] + b[l], Compute A[l] = g(Z[l]) where g is activation function
    """
    raise NotImplementedError
  def derivative(self, Z: np.ndarray) -> np.ndarray:
    """
    Given A[l] = g(Z[l]), Compute dA[l]/dZ[l]
    """
    raise NotImplementedError

class Sigmoid(Function):
  """
  Vectorized implementation of the sigmoid activation function and its derivative
  """
  
  def activation(self, Z: np.ndarray) -> np.ndarray:
    """
    Vectorized implementation of the sigmoid activation function applied to a matrix, returns activation matrix A
    
    For the ijth element of Z, Aij = 1/(1+e^-Zij).
    
    Note: the value Z is clipped to prevent overflow
    Test(s):
    >>> function = Sigmoid()
    >>> nums = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> expected_output = [[0.73105858, 0.88079708, 0.95257413], [0.98201379, 0.99330715, 0.99752738], [0.99908895, 0.99966465, 0.99987661]]
    >>> np.testing.assert_allclose(function.activation(nums), expected_output, rtol=1e-5) is None
    True
    """
    Z = np.clip(Z, -500, 500)
    A = 1 / (1 + np.exp(-Z))
    return A
  
  def derivative(self, Z: np.ndarray) -> np.ndarray:
    """
    Vectorized implementation for finding the gradient of the sigmoid activation function
    with respect to its input matrix Z. ie find dA[l]/dZ[l]
    
    Derivation of the formula:
    
    Consider:
    Z =   [z11, ..., z1n]  and A = σ function, A(z) is the activation matrix for sigmoid function
          [.............]
          [zn1, ..., znn]
          
          
    Consider any ijth entry of Z, zij, Aij = (1+e^-zij)^-1 (*) (Note that Aij is a function of only zij)
    
    <-> e^-zij = 1/Aij - 1 (**)
    
    Taking the derivative of (*) wrt zij on both sizes:
    
        dAij/zij = -1 * [(1 + e^-zij) ^ -2] * - e ^ -zij
      
    <=> dAij/zij =  (1/Aij - 1) * Aij ^2 using (**)

    <=> dAij/zij =  Aij*(1-Aij) as needed.

    (Note that dAij/zkl = 0 if i != k or j != l)
    
    Test(s):
    >>> function = Sigmoid()
    >>> nums = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> expected_output = [[1.96611933e-01, 1.04993585e-01, 4.51766597e-02], [1.76627062e-02, 6.64805667e-03, 2.46650929e-03], [9.10221180e-04, 3.35237671e-04, 1.23379350e-04]]
    >>> np.testing.assert_allclose(function.derivative(nums), expected_output, rtol=1e-5) is None
    True
    """
    return self.activation(Z) * (1 - self.activation(Z))



class ReLU(Function):
  """
  Vectorized implementation of the ReLU activation function and its derivative

  Instance Attributes:
  - Z: a matrix representing a batch of inputs after weights/biases applied but before ReLU
  applied for a given layer
  """

  def activation(self, Z: np.ndarray) -> np.ndarray:
      """
      Vectorized implementation of the relu activation function applied to a matrix, returns activation A
      
      For the ijth element of Z, Aij = max(0, Zij)
      
      Test(s):
      >>> function = ReLU()
      >>> nums = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
      >>> expected_output = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
      >>> np.testing.assert_allclose(function.activation(nums), expected_output, rtol=1e-5) is None
      True
      """
      A = Z * (Z > 0)
      return A
  
  def derivative(self, Z: np.ndarray) -> np.ndarray:
    """
    Vectorized implementation for finding the gradient of the relu activation function with respect to its
    input matrix.
    
    Derivation of the formula:
    
    Consider:
    Z =   [z11, ..., z1n]  and A = relu function, A(z) is the activation matrix for relu function
          [.............]
          [zn1, ..., znn]
          
    Consider any ijth entry of Z, Aij = max(zij, 0) (*)
    
    dAij/zij = 1 if zij > 0, 0 if zij <= 0 as needed
    
    Test:
    >>> function = ReLU()
    >>> nums = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
    >>> expected_output = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    >>> np.testing.assert_allclose(function.derivative(nums), expected_output, rtol=1e-5) is None
    True
    """
    return np.where(Z > 0, 1, 0)


if __name__ == '__main__':
  import doctest
  doctest.testmod()