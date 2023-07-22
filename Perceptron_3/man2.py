# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:36:32 2023

@author: vennapusa manoj
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the line equation (e.g., y = ax + b)
a = 0.5
b = -200

# Generate random data points and assign class labels
np.random.seed(0)
data_points_above = np.random.rand(10, 2) * 1000
data_points_below = np.random.rand(10, 2) * 1000
labels_above = np.sign(a * data_points_above[:, 0] + b - data_points_above[:, 1])
labels_below = np.sign(a * data_points_below[:, 0] + b - data_points_below[:, 1]) * -1

# Concatenate the above and below data points and labels
data_points = np.concatenate((data_points_above, data_points_below))
labels = np.concatenate((labels_above, labels_below))

# Initialize weights and learning rate
weights = np.random.rand(3)
learning_rate = 0.0001

# Create a figure for visualization
fig, ax = plt.subplots()

# Function to update the plot
def update_plot():
    ax.clear()
    ax.scatter(data_points[labels == 1, 0], data_points[labels == 1, 1], c='red', marker='o', label='Class 1')
    ax.scatter(data_points[labels == -1, 0], data_points[labels == -1, 1], c='blue', marker='o', label='Class -1')
    ax.plot([0, 1000], [line_y(0), line_y(1000)], 'g-', label='Current Line')
    ax.legend()
    plt.pause(0.1)

# Function to calculate the line equation
def line_y(x):
    return -(weights[0] + weights[1] * x) / weights[2]

# Function to train the perceptron
def train_perceptron(weights):
    epoch = 0
    misclassified = True

    while misclassified:
        misclassified = False
        misclassified_count = 0

        for i in range(len(data_points)):
            x = np.insert(data_points[i], 0, 1)  # Add bias term
            y = labels[i]
            prediction = np.sign(np.dot(weights, x))

            if prediction != y:
                misclassified = True
                misclassified_count += 1
                weights += learning_rate * y * x

        print(f"Epoch {epoch}: Misclassified = {misclassified_count}")
        update_plot()

        if not misclassified:
            break

        epoch += 1

# Initialize the plot
update_plot()

# Train the perceptron
train_perceptron(weights)

# Keep the plot window open
plt.show()
