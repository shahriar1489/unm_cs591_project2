import numpy as np


class Widrow:
    def __init__(self, features):
        np.random.seed(0)
        self.weight = np.random.rand(features)
        self.bias = 0

    def forward(self, input):
        output = np.dot(self.weight, input) + self.bias

    def fit(self, X, y, epochs=10, lr=1e-6):
        observations, features = X.shape

        # training
        for epoch in epochs:
            total_loss = 0
            for o in range(observations):
                y_hat = self.forward(X[o])
                loss = (y[o], y_hat) ** 2

                self.weights += 2 * lr * loss * X[o]
                self.bias += 2 * lr * loss
                total_loss += loss

            print(f"For Epoch{epoch +1}/ {epochs}, Loss: {total_loss}")
