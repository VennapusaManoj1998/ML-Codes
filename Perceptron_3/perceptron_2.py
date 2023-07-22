import numpy as np
import matplotlib.pyplot as plt


a, b = 0.5, -200


num_points, canvas_size = 20, 1000
points = np.random.rand(num_points, 2) * canvas_size
classes = np.where(points[:, 1] > a * points[:, 0] + b, 1, -1)

weights, learning_rate = np.random.rand(3), 0.0001


def perceptron_learning(x, y, weights):
    weights += learning_rate * (y - np.sign(np.dot(weights, x))) * x
    return weights, int(y != np.sign(np.dot(weights, x)))


epoch, misclassified = 0, num_points
plt.figure(figsize=(8, 8))

while misclassified > 0 and epoch <= 200:
    misclassified = sum(perceptron_learning(np.append(point, 1), label, weights)[1]
                        for point, label in zip(points, classes))
    print('Epoch: {} Missclassified: {}'.format(epoch + 1, misclassified))
    plt.clf()
    plt.plot(points[classes == 1, 0], points[classes == 1, 1], 'ko', fillstyle='none', label='Class 1')
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
