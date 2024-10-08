import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVM:
  def __init__(self,
               learning_rate=0.001,
               lambda_param=0.01,
               n_iters=1000):

    # "Each class should have a constructor which initializes the weights and bias,"
    # ->> to initialize the weights and biases, the dimnensions should be known

    #instances, features = X.shape


    self.lr = learning_rate
    self.lambda_param = lambda_param

    self.n_iters = n_iters
    self.w = None
    self.b = None


  def forward(self, X_):
    #a function forward which computes the output of the ML model based on a given input
    return ( X_.transpose() @ self.w).item() - self.b

  def fit(  self, X, y):
    n_samples, n_features = X.shape
    self.w = np.random.rand(n_features, 1)
    self.b = np.random.rand(1)

    y = np.where(y <= 0, -1, 1)
    loss_list = []

    for _ in range(self.n_iters):
      total_loss = 0 
      for idx, x_i in enumerate(X): #
        x_i = x_i.reshape((-1, 1))
        y_i_pred = self.forward(x_i)
        
        loss = (y[idx] * y_i_pred)[0]
        total_loss += max(0, 1 - loss)
        
        if (loss >= 1):
          self.w -= self.lr * (2 * self.lambda_param * self.w) # updating weight by a factor of regularization term
          self.b -= 0 # not updating the bias

        else: # penalize  the weight and bias  for making error wrt to the weight
          self.w -= self.lr * ( 2 * self.lambda_param * self.w - (x_i * y[idx]) ) # no bias was used in this eqn
          self.b -= self.lr * y[idx]
      avg_loss = total_loss / n_samples
      loss_list.append(avg_loss) 
      print(f"Iteration {_ + 1}: Average Loss: {avg_loss}")

      
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', color='red')
    plt.title("Loss Reduction for Support Vector Machine")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.grid(True) # if we want grid
    plt.show()
  def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)
