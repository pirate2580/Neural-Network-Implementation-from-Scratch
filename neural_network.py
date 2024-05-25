"""
A module for the neural network class
Author: Naoroj Farhan
Date: Thursday, May 23, 2024
"""
import numpy as np
from functions import *
from losses import *

class NN:
    """
    A neural network class
    
    Instance Attributes: 
    structure: a list of tuples for type of the layer as a string and the number of neurons in the layer, supported types for the layer are input, identity, sigmoid, and relu.
                The first element of structure must necessarily be input
                
    bias: A dictionary representing a mapping of the bias vector for each layer of the neural network
    
    weight: A dictionary representing a mapping of the weight matrices for each layer of the neural network
    """
    
    accepted_nonlinearity = ['sigmoid', 'relu', 'none']
    accepted_loss_type = ['MeanSquaredError']
    structure: list[tuple[str, int]]
    bias: dict[str, np.ndarray]
    weight: dict[str, np.ndarray]
    loss: str
        
    def __init__(self, structure: list[tuple[str, int]], loss: str) -> None:
        """
        Initialization method of the neural network,
        raises Error if the first layer isn't 'input' layer and if subsequent layers aren't of type 'none', 'sigmoid' or 'relu'

        >>> test = NN([], 'MeanSquaredError')
        Traceback (most recent call last):
        ...
        ValueError: Cannot pass empty array for structure of neural network
        >>> test = NN([('not input', 5)], 'MeanSquaredError')
        Traceback (most recent call last):
        ...
        ValueError: First layer must be an input layer
        >>> test = NN([('input', 5), ('nonsense', 5), ('more nonsense', 5)], 'MeanSquaredError')
        Traceback (most recent call last):
        ...
        ValueError: the non-linearity function in layer 1 is not accepted
        >>> test = NN([('input', 10),('sigmoid', 5), ('relu', 5)], 'asdf')
        Traceback (most recent call last):
        ...
        ValueError: Invalid loss type: asdf
        >>> test = NN([('input', 10),('sigmoid', 5), ('relu', 5)], 'MeanSquaredError')
        >>> test.weight['W1'].shape
        (5, 10)
        >>> test.bias['b2'].shape
        (5, 1)
        """
        self.structure = structure
        self.loss = loss

        if len(self.structure) == 0:
            raise ValueError('Cannot pass empty array for structure of neural network')
        if self.loss not in self.accepted_loss_type:
            raise ValueError(f'Invalid loss type: {self.loss}')
        if self.structure[0][0] != 'input':
            raise ValueError('First layer must be an input layer')
        for l in range (1, len(structure)):
            if structure[l][0] == 'input':
                raise ValueError('Only first layer can be an input layer')
            if structure[l][0] not in self.accepted_nonlinearity:
                raise ValueError(f'the non-linearity function in layer {l} is not accepted')

        self.bias = {"b" + str(i) : np.random.randn(structure[i][1], 1) for i in range(1, len(structure))}
        self.weight = {"W" + str(i) : np.random.randn(structure[i][1], structure[i - 1][1]) for i in range(1, len(structure))}

    def layer_type(self, layer_type: str) -> Function:
      """
      Helper function to return the type of non-linearity for a given layer if applicable.

      >>> test = NN([('input', 10),('sigmoid', 5), ('relu', 5)], 'MeanSquaredError')
      >>> type(test.layer_type(test.structure[1][0]))
      <class 'functions.Sigmoid'>
      """
      if layer_type == 'sigmoid':
          return Sigmoid()
      elif layer_type == 'relu':
          return ReLU()
      else:
          return None
    
    def loss_type(self) -> Loss:
      """
      Helper function to return the type of loss function specified for a neural network
      """
      if self.loss == 'MeanSquaredError':
          return MeanSquaredError()

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized implementation of the forward propagation algorithm for m training examples where there are m columns representing each training example.
        Each batch must be in a form where each column is a training example and each row is a feature
        it applies the activation function specified in the structure of the initialization method
            - supports ReLU, sigmoid, and identity activation
        
        Returns AL, the activations of the output layer for a matrix comprised of training examples
        
        Equations:
        Z[l] = W[l] * A[l-1] + b[l]
        A[l] = g(Z[l])
        Representation Invariant:
        - X.shape[0] = structure[0][1]

        test(s):
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])   # each row represents a training example, 4x3
        >>> neural_network = NN([('input', 5),('sigmoid', 1)], 'MeanSquaredError')
        >>> neural_network.forward_propagation(X)
        Traceback (most recent call last):
        ...
        ValueError: number of columns of X (number of features): 4, does not match the number of inputs of neural network: 5
        >>> neural_network = NN([('input', 4),('sigmoid', 2)], 'MeanSquaredError')
        >>> neural_network.weight['W1'] = np.array([[1, 0, -1, 0], [1, 0, -1, 0]])
        >>> neural_network.bias['b1'] = np.array([[0, 0, 0], [0, 0, 0]])
        >>> test = neural_network.forward_propagation(X)
        >>> test.shape == (2, 3) # a sigmoid output for each of 3 training examples
        True
        >>> expected_output = [[0.00247262315663, 0.00247262315663, 0.00247262315663], [0.00247262315663, 0.00247262315663, 0.00247262315663]]
        >>> np.testing.assert_allclose(test, expected_output, rtol=1e-5) is None
        True
        """

        if X.shape[0] != self.structure[0][1]:
            raise ValueError(f'number of columns of X (number of features): {X.shape[0]}, does not match the number of inputs of neural network: {self.structure[0][1]}')
        
        A = np.copy(X)
        for l in range (1, len(self.structure)):
            W_l = self.weight["W" + str(l)]
            b_l = self.bias["b" + str(l)]
            # print('asdfasdfsd', W_l.shape, b_l.shape)
            assert W_l.shape[1] == A.shape[0]  # ensures # columns of W_l == # rows of A_(l-1)
            
            Z_l = np.dot(W_l, A) + b_l
            function = self.layer_type(self.structure[l][0])
            if function is not None:
              A = function.activation(Z_l)
           
        return A
    
    def back_propagation(self, X: np.ndarray, Y: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Vectorized implementation of the backpropagation algorithm.
        
        returns a dictionary of the gradients wrt W and B for each layer from a matrix X comprised of all training examples
        
        Representation Invariants: 
        - X.shape[1] is the number of training examples
        """
        # print([i for i in range(len(self.structure))])
        dJ_dW = {f"{l}" : np.random.randn(self.structure[l][1], self.structure[l - 1][1]) for l in range(1, len(self.structure))}
        dJ_dB = {f"{l}" : np.zeros((self.structure[l][1], 1)) for l in range(1, len(self.structure))}
        Z_cache = {}
        A_cache = {}
        
        # forward propagation to fill Z_cache and A_cache for later back propagation steps
        A = np.copy(X)
        A_cache["A0"] = X
        for l in range (1, len(self.structure)):
            W_l = np.copy(self.weight["W" + str(l)])
            b_l = np.copy(self.bias["b" + str(l)])
            
            assert np.shape(W_l)[1] == np.shape(A)[0]  # ensures dimensions align for matrix multiplication
            
            Z_l = np.dot(W_l, A) + b_l
            Z_cache["Z" + str(l)] = np.copy(Z_l)
            
            function = self.layer_type(self.structure[l][0])
            if function is not None:
              A = function.activation(Z_l)
            
            A_cache["A" + str(l)] = np.copy(A)
        
        loss_function = self.loss_type()
        dJ_dAl = loss_function.derivative(A, Y)     # derivative of cost function with respect to prediction, (nl, m) matrix
        
        for l in range (len(self.structure) - 1, 0, -1):
            function = self.layer_type(self.structure[l][0])

            if function is not None:
            # we want cost wrt Z but dA/dZ depends on the activation function used so we account for this
              dJ_dZl = dJ_dAl * function.derivative(Z_cache[f"Z{l}"])
            else:
              dJ_dZl = dJ_dAl

            # I think rigorously dZ_dWl should be a 4d Jacobian tensor of dimension (nl, m, nl, nl-1), but it can be proven to simplify to the formula: dJ_dW[l] = dJ_dZ[l] matrix multiplied by A[l-1].T
            dJ_dW[f"{l}"] = np.copy(np.dot(dJ_dZl, A_cache[f"A{l - 1}"].T))
            dJ_dB[f"{l}"] = np.copy(np.sum(dJ_dZl, axis=1, keepdims=True))  # this is because dZ_dB is just column vector of all 1s
            # ith column of w gets ith row of a
            dJ_dAl = np.dot(dJ_dW[f"{l}"].T, dJ_dZl)
        
        return dJ_dW, dJ_dB
    
    def gradient_descent(self, iterations, learning_rate, X, Y):
        """gradient descent algorithm, takes the dictionary of gradient steps for W and b and increments them for a certain number of iterations
        returns costs np array where the ith value is the model's cost at the ith iteration"""
        learning_curve = np.zeros(iterations)
        loss_function = self.loss_type()
        # print('weights 0th iteration are', neural_network.weight['W1'], neural_network.bias['b1'])
        for itr in range(iterations):
            dJ_dW, dJ_dB = self.back_propagation(X, Y)
            # print(f'gradients {itr}th iteration are: ', dJ_dW, dJ_dB)
            for l in range(1, len(self.structure)):
                # assert self.weight["W" + str(l)].shape == dJ_dW[str(l)].shape
                # assert self.bias["b" + str(l)].shape == dJ_dB[str(l)].shape
                
                self.weight["W" + str(l)] -= learning_rate * dJ_dW[str(l)]
                self.bias["b" + str(l)] -= learning_rate * dJ_dB[str(l)]
            # print(f'weights {itr + 1}th iteration are', neural_network.weight['W1'], neural_network.bias['b1'])
            # print(f'current prediction is: {self.forward_propagation(X)}')
            # print(f'loss at {itr}th iteration is: {loss_function.evaluate(self.forward_propagation(X), Y)}')
            learning_curve[itr] = loss_function.evaluate(self.forward_propagation(X), Y)
        
        return learning_curve


if __name__ == '__main__':
  import doctest
  doctest.testmod()
  import matplotlib.pyplot as plt
  X = np.array([[1, 0, 1], [0, 1, 0], [0, -1, 0], [-1, 0, -1]])
  neural_network = NN([('input', 4),('sigmoid', 1)], 'MeanSquaredError')
  neural_network.weight['W1'] = np.array([[1., 1., 1., 1.]])
  neural_network.bias['b1'] = np.array([[0., 0., 0.]])
  Y = [[1, 0, 1]]
  iters = 100
  # You will see that the cost goes down gradually
  training_curve = neural_network.gradient_descent(iters, 0.1, X, Y)
  plt.plot(np.arange(iters), training_curve)
  plt.title("Learning Curve for Neural Network")
  plt.ylabel("Cost function")
  plt.xlabel("iterations")
  plt.show()