import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
w = 4
b = 5
x = np.random.uniform(-10, 10, 20)
#print(x)
noise = np.random.uniform(-0.1, 0.1, 20)
y = w * x + b
y_noise = y + noise * y
print(x)
print(y)
print(y_noise)


w_random = np.random.uniform(0, 1)
b_random = np.random.uniform(0, 1)
lr = 0.000001
num_epochs = 500
w_list = []
b_list = []
mse_list = []
for epoch in range(num_epochs):
    for i in range(len(x)):
        y_pred = w_random * x[i] + b_random

        error_b = y[i] - y_pred
        error_w = (y[i] - y_pred) * x[i]
        b_random += lr * error_b
        w_random += lr * error_w

    # Loss
    mse = np.mean((y_pred - y_noise[i])**2)

    w_list.append(w_random)
    b_list.append((b_random))
    mse_list.append(mse)
    print(f"Epoch {epoch+1}: MSE = {mse_list[epoch]}")
# Create a figure and axis for the animation
fig, ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Animation')

# Initialize an empty line object
line = Line2D([], [], color= 'r', label='Regression Line')
ax.add_line(line)

# Initialize the plot with the ground truth line
ax.plot(x, y_noise, 'go', 'Data Points')
ax.plot(x, y, 'g-', label='Ground Truth Line')

# Define the update function for the animation
for epoch in range(num_epochs):

    x_line = np.array([-10, 10])
    y_line = w_list[epoch] * x_line + b_list[epoch]
    line.set_data(x_line, y_line)

    # Set the title with the epoch number
    ax.set_title(f'Epoch {epoch+1}')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)
plt.show()