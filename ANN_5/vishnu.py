import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data file
def preprocess_data(filename):
    # Read the data file
    data = pd.read_csv(filename)

    # Normalize the input values
    data.iloc[:, :64] /= 16.0

    # Convert the desired output digit to a target output vector
    tgt = np.zeros((data.shape[0], 4))
    tgt_values = data.iloc[:, 64]

    for i in range(tgt.shape[0]):
        tgt[i, int(tgt_values[i])] = 0.9
        tgt[i, np.where(tgt[i] != 0.9)] = 0.1

    return data, tgt
X, Y = preprocess_data('./data/optdigits-3.tra')
x_test, y_test = preprocess_data('./data/optdigits-3.tes')

x_train = X[:int(0.8 * len(X))]
y_train = Y[:int(0.8* len(Y))]
x_val = X[int(0.8 * len(X)):]
y_val = Y[int(0.8* len(Y)):]
# print(x_train.shape)
# print(y_train.shape)
# print(x_val.shape)
# print(y_val.shape)
# print(x_test.shape)
print(y_test.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test, input_size, hidden_size, output_size, epochs,
                         learning_rate):
    # Initialize weights with random values
    weights1 = np.random.randn(input_size, hidden_size)
    weights2 = np.random.randn(hidden_size, output_size)

    # Initialize biases with zeros
    bias1 = np.zeros((1, hidden_size))
    bias2 = np.zeros((1, output_size))

    # Variables to store MSE values
    training_mse = []
    validation_mse = []

    for epoch in range(epochs):
        # Forward propagation
        hidden_layer = sigmoid(np.dot(X_train, weights1) + bias1)
        output_layer = sigmoid(np.dot(hidden_layer, weights2) + bias2)

        # Calculate and store the MSE for training and validation sets every 10 epochs
        if epoch % 10 == 0:
            training_mse.append(np.mean(np.square(y_train - output_layer)))

            hidden_layer_val = sigmoid(np.dot(X_val, weights1) + bias1)
            output_layer_val = sigmoid(np.dot(hidden_layer_val, weights2) + bias2)
            validation_mse.append(np.mean(np.square(y_val - output_layer_val)))

        # Backpropagation
        delta_output = (y_train - output_layer) * sigmoid_derivative(output_layer)
        delta_hidden = delta_output.dot(weights2.T) * sigmoid_derivative(hidden_layer)

        # Update weights and biases
        weights2 += hidden_layer.T.dot(delta_output) * learning_rate
        bias2 += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        weights1 += X_train.T.dot(delta_hidden) * learning_rate
        bias1 += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    # Plot the MSE values as a training curve
    epochs_range = range(0, epochs, 10)
    plt.plot(epochs_range, training_mse, label='Training Set')
    plt.plot(epochs_range, validation_mse, label='Validation Set')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Training plot and Validation plot')
    plt.legend()
    plt.show()

    # Testing
    hidden_layer_test = sigmoid(np.dot(X_test, weights1) + bias1)
    output_layer_test = sigmoid(np.dot(hidden_layer_test, weights2) + bias2)
    predicted_labels = np.argmax(output_layer_test, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.sum(predicted_labels == true_labels) / len(true_labels) * 100
    print(f"Test Set Accuracy: {accuracy:.2f}%")


# Train the neural network and test
train_neural_network(x_train, y_train, x_val, y_val, x_test, y_test, x_train.shape[1], 50, y_train.shape[1], 1000, 0.01)