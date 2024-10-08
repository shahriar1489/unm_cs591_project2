import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:
    def __init__(self, learning_rate, epochs, input_size):
        np.random.seed(0)
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.random.randn(input_size) * 0.001
        self.bias = 0.01

    def sigmoid(self, z):
        sigmoid_val = 1 / (1 + np.exp(-z))
        return sigmoid_val

    def forward(self, input):
        z = np.dot(input, self.weights) + self.bias
        sigmoid_val = self.sigmoid(z)
        return sigmoid_val

    def calculate_loss(self, true_labels, predicted_labels):
        """
        Calculate the custom loss function from the book:
        LC = -sum(log(y_i / 2 - 0.5 + predicted_labels))
        
        true_labels: Actual class labels (0 or 1)
        predicted_labels: Sigmoid outputs (predicted probabilities)
        """
        # Adjust the true labels: y_i / 2 - 0.5
        adjusted_labels = (true_labels / 2) - 0.5

        # Compute the loss for all examples
        epsilon = 1e-15
        value = np.abs(adjusted_labels + predicted_labels) + epsilon
        value = np.log(value)
        loss = -np.mean(value)
        
        return loss
    
    def calculate_gradients(self, input, labels, predictions):
        """
        Calculate gradients for weights and bias using the custom loss function gradient from the book.
        
        input: The input data (n_samples x n_features)
        labels: The true labels (n_samples)
        predictions: The predicted probabilities (n_samples)
        
        Returns the gradient for weights and bias.
        """
        n_samples = input.shape[0]
        adjusted_labels = (labels / 2) - 0.5  # Adjust the labels as per the book's formula

        # Initialize gradients
        gradient_weights = np.zeros_like(self.weights)
        gradient_bias = 0

        # Loop through each sample to calculate individual gradients
        for i in range(n_samples):
            Xi = input[i]
            yi = adjusted_labels[i]
            pred = predictions[i]

            # Calculate the exp term: exp(y_i * (W^T * X_i))
            exp_term = np.exp(yi * np.dot(self.weights, Xi))  # exp(y_i W^T X_i)
            denominator = 1 + exp_term  # Denominator: 1 + exp(y_i W^T X_i)

            # Update gradients
            gradient_weights += (yi * Xi) / denominator  # Gradient w.r.t weights
            gradient_bias += yi / denominator  # Gradient w.r.t bias

        # Take the average over the number of samples
        gradient_weights /= n_samples
        gradient_bias /= n_samples

        return gradient_weights, gradient_bias

    def fit(self, input, labels):
        n_rows, n_features = input.shape
        
        loss_list = []
        for i in range(self.epochs):
            predicted_labels = self.forward(input)
            loss = self.calculate_loss(labels, predicted_labels)
            loss_list.append(loss)

            print(f'Epoch {i+1}, Loss: {loss}')

            gradient_weights, gradient_bias = self.calculate_gradients(input, labels, predicted_labels)

            self.weights += self.lr * gradient_weights
            self.bias += self.lr * gradient_bias
        plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', color='red')
        plt.title("Loss Reduction for Logistic Regression")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.grid(True) # if we want grid
        plt.show()
    
    def predict(self, input):
        """
        Predict binary class labels (0 or 1) for the given input.
        
        input: The input data (n_samples x n_features)
        
        Returns binary class labels (0 or 1) for each sample.
        """
        predicted_probs = self.forward(input)
        
        # Step 2: Apply a threshold to convert probabilities to binary labels
        predicted_labels = (predicted_probs >= 0.5).astype(int)
        
        return predicted_labels
            

