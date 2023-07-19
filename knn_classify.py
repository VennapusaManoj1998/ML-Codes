# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 05:39:55 2023

@author: manoj kumar reddy, vennapusa
M#: M03484564
"""
import sys
import numpy as np


def normalize(data):
    # Normalize the data by subtracting the mean and dividing by the standard deviation
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return np.sqrt(np.sum((point1 - point2) ** 2))


def knn_classify(training_file, test_file, k):
    # Load training data
    training_data = np.loadtxt(training_file)
    training_labels = training_data[:, -1]
    training_features = normalize(training_data[:, :-1])

    # Load test data
    test_data = np.loadtxt(test_file)
    test_labels = test_data[:, -1]
    test_features = normalize(test_data[:, :-1])

    num_test = test_data.shape[0]
    correct = 0

    for i in range(num_test):
        distances = []
        for j in range(training_data.shape[0]):
            # Calculate the Euclidean distance between the test point and each training point
            dist = euclidean_distance(test_features[i], training_features[j])
            distances.append((dist, training_labels[j]))

        distances.sort(key=lambda x: x[0])  # Sort distances in ascending order

        # Get the k nearest neighbors
        neighbors = [x[1] for x in distances[:k]]

        # Count the occurrences of each class in the neighbors
        counts = np.bincount(neighbors)
        predicted_class = np.argmax(counts)
        true_class = test_labels[i]

        # Calculate accuracy
        if predicted_class == true_class:
            # Check if the predicted class is correct
            if len(np.unique(neighbors)) == 1:
                # If there were no ties in the classification result
                accuracy = 1.0
            else:
                # If there were ties in the classification result
                accuracy = 1.0 / np.sum(neighbors == true_class)
        else:
            accuracy = 0.0
        
        correct += accuracy


        # Print the classification result for the current test point
        print("ID=%5d, predicted=%3d, true=%3d" %
              (i, predicted_class, int(test_labels[i])))

    classification_accuracy = correct / num_test
    # Print the overall classification accuracy
    print("classification accuracy=%6.4lf" % classification_accuracy)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        # Check if the number of arguments is correct
        print("Usage: python knn_classify.py <training_file> <test_file> <k>")
    else:
        training_file = sys.argv[1]
        test_file = sys.argv[2]
        k = int(sys.argv[3])
        # Call the knn_classify function with the provided arguments
        knn_classify(training_file, test_file, k)
