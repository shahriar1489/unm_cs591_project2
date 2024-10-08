import numpy as np
import matplotlib.pyplot as plt


class Widrow:
    def __init__(self, features):
        np.random.seed(0)
        # Initialization of the weight and bias.
        self.weights = np.random.rand(features) * 0.05
        self.bias = 0.01

    def forward(self, input):
        # forward to get the output from the model
        output = np.dot(input, self.weights) + self.bias
        output = np.where(output >= 0.5, 1, 0)
        return output

    def fit(self, X, y, epochs, lr):
        observations, features = X.shape
        loss_list = []

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
            loss_list.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

        plt.plot(range(1, len(loss_list) + 1), loss_list, marker="o", color="red")
        plt.title("Loss Reduction for Widrow Hoff Model.")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.grid(True)  # if we want grid
        plt.show()
