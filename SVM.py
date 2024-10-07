# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:36:13 2024

@author: Odysseus Valdez
CS 491 Assignment 2 Linear Support Vector Machine (SVM)
"""

import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    
    def forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        y2 = np.where(y <= 0, -1, 1)

        # Training using gradient descent
        for _ in range(self.iterations):
            for i, x_index in enumerate(X):
                # Condition for loss calculation
                condition = y2[i] * (np.dot(x_index, self.weights) - self.bias) >= 1
                if condition:
                    # Correct classification
                    self.weights -= self.learning_rate * (2 * 1/self.iterations * self.weights)
                else:
                    # Incorrect classification
                    self.weights -= self.learning_rate * (2 * 1/self.iterations * self.weights - np.dot(x_index, y2[i]))
                    self.bias -= self.learning_rate * y2[i]

    def predict(self, X):
        linear_output = self.forward(X)
        return np.where(linear_output >= 0, 1, 0)

# Example Usage
if __name__ == "__main__":

    X = np.array([[1, 2], [2, 3], [3, 3], [5, 1], [5, 4], [6, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])

    svm = SVM(learning_rate=0.01, iterations=1000)
    svm.fit(X, y)

    predictions = svm.predict(X)
    print("Predictions:", predictions)

        
        