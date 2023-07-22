import numpy as np
import matplotlib.pyplot as plt


a = 0.5
b = -200

num_points = 20
canvas_size = 1000
points = np.random.rand(num_points, 2) * canvas_size
classes = np.where(points[:, 1] > a * points[:, 0] + b, 1, -1)

plt.figure(figsize=(8, 8))
plt.plot(points[classes == 1, 0], points[classes == 1, 1], 'ko', fillstyle='none', label='Class 1')
plt.plot(points[classes == -1, 0], points[classes == -1, 1], 'ko', label='Class -1')
plt.plot([0, canvas_size], [b, a * canvas_size + b], 'g', label='Line')
plt.legend()
plt.xlim([0, canvas_size])
plt.ylim([0, canvas_size])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Perceptron Linear Separability')
plt.show()

a = 0.5
b = -200

num_points = 20
canvas_size = 1000
points = np.random.rand(num_points, 2) * canvas_size
classes = np.where(points[:, 1] > a * points[:, 0] + b, 1, -1)


weights = np.random.rand(3)
learning_rate = 0.0001


def perceptron_learning(x, y, weights):
    prediction = np.sign(np.dot(weights, x))
    error = y - prediction
    weights += learning_rate * error * x
    return weights, error


epoch = 0
misclassified = num_points
plt.figure(figsize=(8, 8))

while misclassified > 0 and epoch <= 200:
    misclassified = 0

    for point, label in zip(points, classes):

        x = np.append(point, 1)


        weights, error = perceptron_learning(x, label, weights)

        if error != 0:
            misclassified += 1
    print('Epoch: {} Missclassified: {}'.format(epoch + 1, misclassified))
 
    plt.clf()
    plt.plot(points[classes == 1, 0], points[classes == 1, 1], 'ko',fillstyle='none', label='Class 1')
    plt.plot(points[classes == -1, 0], points[classes == -1, 1], 'ko', label='Class -1')
    plt.plot([0, canvas_size], [-weights[2] / weights[1], (-weights[2] - weights[0] * canvas_size) / weights[1]], 'g', label='Line')
    plt.legend()
    plt.xlim([0, canvas_size])
    plt.ylim([0, canvas_size])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptron Linear Separability (Epoch: {})'.format(epoch))
    plt.pause(0.1)

    epoch += 1

print('Training completed. Total epochs:', epoch - 1)
plt.show()
