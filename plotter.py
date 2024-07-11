import numpy as np
import matplotlib.pyplot as plt


def training_function(x):
    return np.where(x < 0, (x + 2) ** 4, 64 * ((x - 0.5) ** 2))


def testing_function(x):
    # return np.where(x < -1, (x + 3) ** 2, 64 * ((x - 1.5) ** 4))
    return np.where(x < -0.3, (x + 2.3) ** 4, 64 * ((x - 0.2) ** 2))


x = np.linspace(-3, 1, 1000)

y_train = training_function(x)
y_test = testing_function(x)

fig, ax = plt.subplots(figsize=(6, 3.6))

ax.plot(x, y_train, label="train function", color="black")
ax.plot(x, y_test, label="test function", color="blue", linestyle="dashed")

# Mark the flat and sharp minima
flat_min = -2
sharp_min = 0.5

ax.scatter(
    [flat_min, sharp_min],
    [training_function(flat_min), training_function(sharp_min)],
    color="black",
)
ax.annotate(
    "flat minimum",
    xy=(flat_min, training_function(flat_min)),
    xytext=(flat_min, training_function(flat_min) - 2),
    arrowprops=dict(facecolor="black", shrink=0.05),
    fontsize=12,
    ha="center",
)
ax.annotate(
    "sharp minimum",
    xy=(sharp_min, training_function(sharp_min)),
    xytext=(sharp_min, training_function(sharp_min) - 2),
    arrowprops=dict(facecolor="black", shrink=0.05),
    fontsize=12,
    ha="center",
)

# Add red bars from minima of training function to testing function
ax.vlines(
    flat_min,
    training_function(flat_min),
    testing_function(flat_min),
    color="red",
    linewidth=5,
)
ax.vlines(
    sharp_min,
    training_function(sharp_min),
    testing_function(sharp_min),
    color="red",
    linewidth=5,
)

# Set labels and title
ax.set_xlabel("parameters (weights)", fontsize=11)
ax.set_ylabel("f(x)", fontsize=11)
ax.set_title(
    "Flat vs. Sharp Minima under Distributional Shift (intuition)",
    fontsize=12,
)

# Remove the axis numbers
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim([0, 16.5])

# Add legend
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
