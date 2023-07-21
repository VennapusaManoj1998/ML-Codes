# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:53:54 2023

@author: vennapusa manoj
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate points in the training set
def generate_data():
    np.random.seed(0)
    x = np.linspace(0, 10, 20)
    y = 2 * x + 3 + np.random.uniform(-0.1 * (2 * x + 3), 0.1 * (2 * x + 3), size=x.shape)
    return x, y

# Step 2: Implement linear regression with gradient descent
def linear_regression(x, y, learning_rate, epochs):
    n = len(x)
    w = np.random.rand()
    b = np.random.rand()
    mse_values = []

    for epoch in range(epochs):

        y_pred = w * x + b
        err_b = np.sum(y_pred - y) / n
        err_w = np.sum((y_pred - y) * x) / n
        
        b += learning_rate * err_b
        w += learning_rate * err_w
        
        mse = np.mean((y_pred - y)**2)
        mse_values.append(mse)
        
        # Visualize the line
        plt.figure()
        plt.scatter(x, y, color='black', label='Data Points')
        plt.plot(x, y_pred, color='yellow', label='Regression Line')
        plt.plot(x, 2*x + 3, color= 'green',label='Truth Line')
        plt.title(f'Epoch {epoch+1}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()
        
        print(f"Epoch {epoch+1}, MSE = {mse_values[epoch]}")
    return w, b, mse_values


if __name__ == '__main__':
    
    # Generate training data
    x_train, y_train = generate_data()

    # Perform linear regression
    learning_rate = 0.000001
    epochs = 500
    w_final, b_final, mse_values = linear_regression(x_train, y_train, learning_rate, epochs)

    # Print the final weights and bias
    print('Final weights:', w_final)
    print('Final bias:', b_final)

    # Plot the mean square error over epochs
    plt.figure()
    plt.Mean Square Error over Epochs')plot(range(1, epochs+1), mse_values)
    plt.title('
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()
    
    
    