import matplotlib.pyplot as plt
import numpy as np

basic_data = np.load("classify_digits_results.npy")

basic_epochs, basic_accuracy = basic_data.T

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(basic_epochs, basic_accuracy, label="Basic Model", marker="o")
ax.set_title("Mnist Classifier Comparison")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True)
plt.show()
