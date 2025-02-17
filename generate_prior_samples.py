import numpy as np

# Number of random datasets to generate
number_of_samples = 100000

# Define the parameter ranges for the pendulum simulation
# Each row corresponds to [g, L, m, k, theta0, omega0]
low_values = np.array([9.8, 0.5, 0.5, 0.8, -0.5 * np.pi, -0.01])
high_values = np.array([9.82, 1.5, 1.5, 1.2, 0.5 * np.pi, 0.01])

# Generate a matrix of random datasets using a uniform distribution
param_matrix = np.random.uniform(low=low_values, high=high_values, size=(number_of_samples, 6))

np.save("data/uniform_samples.npy", param_matrix)

"""
If ever needed, here's a way to write data by chunks

# Example of saving data in chunks (if needed)
chunk_size = 10000  # Define the chunk size
for i in range(0, number_of_samples, chunk_size):
    param_chunk = np.random.uniform(low=low_values, high=high_values, size=(chunk_size, 6))
    # Saving the first chunk normally, and then appending
    if i == 0:
        np.save("data/datasets/uniform_samples.npy", param_chunk)
    else:
        with open("data/datasets/uniform_samples.npy", 'ab') as f:
            np.save(f, param_chunk)
"""