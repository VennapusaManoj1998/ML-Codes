# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:36:16 2023

@author: vennapusa manoj
"""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
def normalize_data(data):
    return data / 16.0

def convert_to_target_vector(label):
    target = np.zeros(4)
    target[label] = 0.9
    target[np.where(target == 0)] = 0.1
    return target

def preprocess_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=int)
    inputs = normalize_data(data[:, :-1])
    targets = np.array([convert_to_target_vector(label) for label in data[:, -1]])
    return inputs, targets

# Step 2: Network Architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.bias1 = np.random.rand(hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias2 = np.random.rand(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs):
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights2) + self.bias2)

    def backpropagation(self, inputs, targets, learning_rate):
        error = targets - self.output
        output_delta = error * self.output * (1 - self.output)
        hidden_delta = np.dot(output_delta, self.weights2.T) * self.hidden_output * (1 - self.hidden_output)

        self.weights2 += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias2 += learning_rate * np.sum(output_delta, axis=0)
        self.weights1 += learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias1 += learning_rate * np.sum(hidden_delta, axis=0)
        
    def predict(self, inputs):
        hidden_output = self.sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        output = self.sigmoid(np.dot(hidden_output, self.weights2) + self.bias2)
        return output

# Step 3: Training and Validation
def compute_mse(targets, outputs):
    return np.mean((targets - outputs) ** 2)

def train_network(network, inputs, targets, learning_rate, epochs, validation_split):
    train_size = int(validation_split * len(inputs))
    train_inputs, train_targets = inputs[:train_size], targets[:train_size]
    val_inputs, val_targets = inputs[train_size:], targets[train_size:]

    train_errors, val_errors = [], []

    for epoch in range(epochs):
        network.feedforward(train_inputs)
        network.backpropagation(train_inputs, train_targets, learning_rate)

        if epoch % 10 == 0:
            train_outputs = network.output
            train_error = compute_mse(train_targets, train_outputs)
            train_errors.append(train_error)

            val_outputs = network.predict(val_inputs)
            val_error = compute_mse(val_targets, val_outputs)
            val_errors.append(val_error)

    return train_errors, val_errors

# Step 4: Plotting
def plot_training_curve(train_errors, val_errors):
    epochs = np.arange(0, len(train_errors) * 10, 10)
    plt.plot(epochs, train_errors, label='Training Set')
    plt.plot(epochs, val_errors, label='Validation Set')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Training Curve')
    plt.legend()
    plt.show()

# Step 5: Testing
def test_network(network, inputs, targets):
    network.feedforward(inputs)
    outputs = np.argmax(network.output, axis=1)
    targets = np.argmax(targets, axis=1)
    accuracy = np.mean(outputs == targets) * 100
    return accuracy

# Step 6: Experiments with Varying Numbers of Hidden Units
def run_experiment(hidden_units):
    inputs, targets = preprocess_data(r'C:\Users\venna\OneDrive\Desktop\Machine Learning\ANN_5\data\data\optdigits-3.tra')

    network = NeuralNetwork(input_size=64, hidden_size=hidden_units, output_size=4)
    train_errors, val_errors = train_network(network, inputs, targets, learning_rate=0.1, epochs=200, validation_split=0.2)
    plot_training_curve(train_errors, val_errors)

    test_inputs, test_targets = preprocess_data(r'C:\Users\venna\OneDrive\Desktop\Machine Learning\ANN_5\data\data\optdigits-3.tes')
    test_accuracy = test_network(network, test_inputs, test_targets)
    print(f'Test Accuracy with {hidden_units} hidden units: {test_accuracy:.2f}%')

run_experiment(hidden_units=5)
run_experiment(hidden_units=50)
