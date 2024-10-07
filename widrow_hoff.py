import numpy as np


class Widrow:
    def __init__(self, features):
        np.random.seed(0)
        # Initialization of the weight and bias.
        self.weights = np.random.rand(features) * 0.05
        self.bias = 0

    def forward(self, input):
        # forward to get the output from the model
        output = np.dot(input, self.weights) + self.bias
        output = np.where(output >= 0.5, 1, 0)
        return output

    def fit(self, X, y, epochs=10, lr=1e-12):
        observations, features = X.shape

        # training
        for epoch in range(epochs):
            total_loss = 0
            for o in range(observations):
                y_hat = np.dot(X[o], self.weights) + self.bias

                # Error
                error = y[o] - y_hat
                loss = error**2

                # weights and bias  update
                self.weights += lr * error * X[o]
                self.bias += lr * error

                # squared error for loss
                total_loss += loss

            # average loss
            avg_loss = total_loss / observations
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
