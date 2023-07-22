# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:51:49 2023

@author: vennapusa manoj
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the line equation (e.g., y = ax + b)
a = 1
b = 100

# Generate random data points and assign class labels
np.random.seed(0)
data_points = np.random.rand(20, 2) * 1000
labels = np.sign(a * data_points[:, 0] + b - data_points[:, 1])

# Initialize weights and learning rate
weights = np.random.rand(3)
learning_rate = 0.0001


# Function to calculate the line equation
def line_y(x):
    return  (-weights[2] + weights[0] * x) / weights[1]

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
        plt.clf()
        plt.scatter(data_points[labels == 1, 0], data_points[labels == 1, 1], c='red', marker='o', label='Class 1')
        plt.scatter(data_points[labels == -1, 0], data_points[labels == -1, 1], c='blue', marker='o', label='Class -1')
        plt.plot([0, 1000], [line_y(0), line_y(1000)], 'g-', label='Current Line')
        plt.legend()
        plt.title(f'Perceptron Linear Separability epoch: {epoch}')
        plt.pause(0.1)
        
        plt.show()

        if not misclassified:
            print('Total number of trainings done are: ', epoch)
            break

        epoch += 1

# Train the perceptron
train_perceptron(weights)

