# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:36:13 2024

@author: Odysseus Valdez
CS 491 Assignment 2 Linear Support Vector Machine (SVM)
"""


import numpy as np
import matplotlib.pyplot as plt


class Weston_Watkins:
    def __init__(self, input_feature, lr=0.01, epochs=1000, num_classes=3):
        np.random.seed(0)
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes

        self.weights = np.random.randn(num_classes, input_feature)
        self.bias = np.ones(num_classes)

    def forward(self, X):
        return np.dot(X, self.weights.T)

    def fit(self, X, y):
        samples, features = X.shape

        num_classes = self.num_classes
        loss_list = []

        # Training using gradient descent
        for epoch in range(self.epochs):
            # making gradient to zero like model.zero_grad before each epoch to aviod gradient accumulation.
            avg_loss = 0.0
            weight_gradient = np.zeros(self.weights.shape)
            bias_gradient = np.zeros(self.bias.shape)

            for i, x_index in enumerate(X):
                logits = self.forward(X[i].reshape(1, -1))[0]
                label = y[i]
                label_logits = logits[label]


                for r in range(num_classes):
                    if r == label:
                        # r = ci
                        class_sum = sum(
                            1
                            for j in range(num_classes)
                            if j != label and logits[j] - label_logits + 1 > 0
                        )
                        weight_gradient[r, :] -= x_index * class_sum
                        bias_gradient[r] -= class_sum

                    else:
                        # r != ci
                        margin = logits[r] - label_logits + 1
                        if margin > 0:
                            avg_loss += margin
                            weight_gradient[r, :] += x_index
                            bias_gradient[label] -= 1

            # Average loss
            avg_loss /= samples
            weight_gradient /= samples
            self.weights -= self.lr * weight_gradient
            loss_list.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.3f}")

        plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', color='red')
        plt.title("Loss Reduction for  Weston Watkins Support Vector Machine")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.grid(True) # if we want grid
        plt.show()

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)
