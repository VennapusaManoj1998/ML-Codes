import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data file
data = pd.read_csv(r'C:\Users\venna\OneDrive\Desktop\Machine Learning\ANN_5\data\data\optdigits-3.tra')
data_test = pd.read_csv(r'C:\Users\venna\OneDrive\Desktop\Machine Learning\ANN_5\data\data\optdigits-3.tes')
# Normalize the input values
x_valu = data.iloc[:, :64] / 16.0
x_test = data_test.iloc[:, :64] / 16.0
# for train and val data
target_output = np.zeros((data.shape[0], 4))
target_digit = data.iloc[:, 64]

for i in range(target_output.shape[0]):
    target_output[i, int(target_digit[i])] = 0.9
    target_output[i, np.where(target_output[i] != 0.9)] = 0.1
#For Test data

test_y = np.zeros((data_test.shape[0], 4))
target_digit_t = data_test.iloc[:, 64]

for i in range(test_y.shape[0]):
    test_y[i, int(target_digit_t[i])] = 0.9
    test_y[i, np.where(test_y[i] != 0.9)] = 0.1

X_train = x_valu[:int(0.8 * len(x_valu))]
y_train = target_output[:int(0.8* len(target_output))]
X_val = x_valu[int(0.8 * len(x_valu)):]
y_val = target_output[int(0.8* len(target_output)):]
print(X_train.shape)
print(y_val.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)

        # Initialize biases with zeros
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))
        self.training_mse = []
        self.validation_mse = []

    def forward(self, X):
        self.hidden_layer = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
        return self.output_layer

    def backward(self, X, y, learning_rate):
        # Backpropagation
        delta_output = (y - self.output_layer) * sigmoid_derivative(self.output_layer)
        delta_hidden = delta_output.dot(self.weights2.T) * sigmoid_derivative(self.hidden_layer)

        # Update weights and biases
        self.weights2 += self.hidden_layer.T.dot(delta_output) * learning_rate
        self.bias2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights1 += X.T.dot(delta_hidden) * learning_rate
        self.bias1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, y_train, learning_rate)

            # Calculate and store the MSE for training and validation sets every 10 epochs
            if epoch % 10 == 0:
                training_mse = np.mean(np.square(y_train - output))
                self.training_mse.append(training_mse)

                validation_output = self.forward(X_val)
                validation_mse = np.mean(np.square(y_val - validation_output))
                self.validation_mse.append(validation_mse)

            # Print the progress
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Training MSE: {training_mse:.6f} - Validation MSE: {validation_mse:.6f}")

        # Plot the MSE values as a training curve
        epochs_range = range(0, epochs, 10)
        plt.plot(epochs_range, self.training_mse, label='Training Set')
        plt.plot(epochs_range, self.validation_mse, label='Validation Set')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Training Curve')
        plt.legend()
        plt.show()

    def test(self, X_test, y_test):
        output = self.forward(X_test)
        predicted_labels = np.argmax(output, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.sum(predicted_labels == true_labels) / len(true_labels) * 100
        print(f"Test Set Accuracy: {accuracy:.2f}%")

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 5
output_size = y_train.shape[1]
learning_rate = 0.01
epochs = 1000

# Create and train the neural network
model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train, y_train, X_val, y_val, epochs, learning_rate)
model.test(x_test, test_y)
