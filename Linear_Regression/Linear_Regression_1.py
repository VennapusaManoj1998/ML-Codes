import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
w = 2
b = 3
x_train = np.random.uniform(-10, 10, 20)
#print(x)
def calculate_y(x, w, b):
    y = w * x + b
    noise = np.random.uniform(-0.1 * y, 0.1 * y)
    #print(noise)
    return y + noise
y_train = calculate_y(x_train, w, b)
y_train_true = w * x_train + b


# Assignment Part 2


w_ran = np.random.uniform(0, 1)
b_ran = np.random.uniform(0, 1)
learning_rate = 0.000001
num_epochs = 500
w_list = []
b_list = []
mse_list = []
for epoch in range(num_epochs):
    for i in range(len(x_train)):
        y_pred = w_ran * x_train[i] + b_ran

        error_b = y_train[i] - y_pred
        error_w = (y_train[i] - y_pred) * x_train[i]
        b_ran += learning_rate * error_b
        w_ran += learning_rate * error_w

    # Loss
    mse = np.mean((y_pred - y_train[i]) ** 2)

    w_list.append(w_ran)
    b_list.append((b_ran))
    mse_list.append(mse)
    print(f"Epoch {epoch+1}: MSE = {mse_list[epoch]}")
# Create a figure and axis for the animation
fig, ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Animation')

# Initialize an empty line object
line, = ax.plot([], [], 'r-', label='Regression Line')

# Initialize the plot with the ground truth line
ax.plot(x_train, y_train, 'go', 'Data Points')
ax.plot(x_train, y_train_true, 'g-', label='Ground Truth Line')

# Define the update function for the animation
def update(frame):
    # Clear the current line
    line.set_data([], [])

    # Plot the current line represented by the weights
    x_line = np.array([-10, 10])
    y_line = w_list[frame] * x_line + b_list[frame]
    line.set_data(x_line, y_line)

    # Set the title with the epoch number
    ax.set_title(f'Epoch {frame+1}')

    return line,

# Create the animation using FuncAnimation
animation = FuncAnimation(fig, update, frames=num_epochs, interval=100)

# Display the animation
plt.legend()
plt.grid(True)
plt.show()