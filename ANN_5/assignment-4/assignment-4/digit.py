import numpy as np
import matplotlib.pyplot as plt
import re

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.W1) + self.b1)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.W2) + self.b2)
        return self.output_layer
    
    def backward(self, inputs, targets, learning_rate):
        output_error = (self.output_layer - targets) * (self.output_layer * (1 - self.output_layer))
        hidden_error = np.dot(output_error, self.W2.T) * (self.hidden_layer * (1 - self.hidden_layer))
        
        self.W2 -= learning_rate * np.dot(self.hidden_layer.T, output_error)
        self.b2 -= learning_rate * np.sum(output_error, axis=0)
        self.W1 -= learning_rate * np.dot(inputs.T, hidden_error)
        self.b1 -= learning_rate * np.sum(hidden_error, axis=0)
    
    def train(self, inputs, targets, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            output = self.forward(inputs)
            self.backward(inputs, targets, learning_rate)
    
    def predict(self, inputs):
        return self.forward(inputs)

def load_dataset(file):
    data = np.loadtxt(file, delimiter=',')
    inputs = data[:, :64] / 16.0  # Normalize input values to [0, 1]
    targets = np.zeros((data.shape[0], 4))
    targets[np.arange(data.shape[0]), data[:, -1].astype(int)] = 0.9  # Convert output digit to target vector
    return inputs, targets

def extract_class_labels(names_file):
    with open(names_file, "r") as f:
        content = f.read()
        match = re.search(r"0\s+Classes\s+0\s+1\s+2\s+3", content)
        if match:
            return [int(label) for label in match.group(0).split()[2:]]
    return []

# Step 1: Load the training and test datasets
train_file = "optdigits-3.tra"
test_file = "optdigits-3.tes"
train_inputs, train_targets = load_dataset(train_file)
test_inputs, test_targets = load_dataset(test_file)

# Step 2: Split the training set into training and validation sets (80% - training, 20% - validation)
val_size = int(train_inputs.shape[0] * 0.2)
validation_inputs = train_inputs[-val_size:]
validation_targets = train_targets[-val_size:]
train_inputs = train_inputs[:-val_size]
train_targets = train_targets[:-val_size]

# Step 3: Set hyperparameters and training settings
num_epochs = 100
hidden_units = [5, 50]  # Number of hidden units for experiments
learning_rates = [0.01, 0.001]  # Updated: Adjust the learning rates for experiments

# Step 4: Training and evaluation
for num_hidden_units in hidden_units:
    for learning_rate in learning_rates:
        # Initialize the neural network with the current number of hidden units
        network = NeuralNetwork(input_size=64, hidden_size=num_hidden_units, output_size=4)

        # Initialize lists to store MSE values for training and validation sets
        train_mse_values = []
        val_mse_values = []

        for epoch in range(num_epochs):
            # Perform forward propagation, backpropagation, and weight updates
            network.train(train_inputs, train_targets, num_epochs=1, learning_rate=learning_rate)

            # Calculate the MSE on the training set after every 10 epochs
            if (epoch + 1) % 10 == 0:
                # Compute MSE on the training set
                train_predictions = network.predict(train_inputs)
                train_errors = train_targets - train_predictions
                train_mse = np.mean(np.square(train_errors))
                train_mse_values.append(train_mse)

                # Compute MSE on the validation set
                val_predictions = network.predict(validation_inputs)
                val_errors = validation_targets - val_predictions
                val_mse = np.mean(np.square(val_errors))
                val_mse_values.append(val_mse)

        # Plot the training and validation curves
        plt.plot(range(10, num_epochs + 1, 10), train_mse_values, label="Training")
        plt.plot(range(10, num_epochs + 1, 10), val_mse_values, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(f"Training and Validation Curves (Hidden Units: {num_hidden_units}, Learning Rate: {learning_rate})")
        plt.legend()
        plt.show()

        # Test the performance on the test set
        test_predictions = network.predict(test_inputs)
        test_outputs = np.argmax(test_predictions, axis=1)
        test_labels = np.argmax(test_targets, axis=1)
        test_accuracy = (np.mean(test_outputs == test_labels)) * 100
        print(f"Test Set Accuracy (Hidden Units: {num_hidden_units}, Learning Rate: {learning_rate}): {test_accuracy:.2f}%")
