import logging
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from pendulum import DampedPendulum

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def simulate_damped_pendulum(sample, index):
    """
    Simulates the motion of a damped pendulum for a given set of parameters.

    Parameters:
    - sample: array-like, containing [g, L, m, k, theta0, omega0]
    - index: integer, representing the sample index

    Returns:
    - index: sample index for correct ordering
    - solution: concatenated array of time values and angular positions
    """
    g, L, m, k, theta0, omega0 = sample
    pendulum = DampedPendulum(g=g, L=L, m=m, k=k)
    t, y = pendulum.solve(theta0=theta0, omega0=omega0)
    return index, np.concatenate((t, y[0]))


def create_dataset(filename):
    """
    Loads sample parameters from a file and computes the corresponding pendulum simulations in parallel.

    Parameters:
    - filename: string, path to the input file containing sample parameters

    Returns:
    - results: numpy array containing simulation results for all datasets
    """

    # Load dataset datasets
    samples = np.load(filename)
    num_samples = samples.shape[0]

    logging.info(f"Solving {num_samples} ODEs from {filename}")

    # Determine the output size by running a single test simulation
    reference_pendulum = DampedPendulum()
    t, _ = reference_pendulum.solve()
    output_size = 2 * len(t)

    # Initialize results array
    results = np.empty((num_samples, output_size), dtype=np.float32)

    # Execute simulations in parallel
    simulations = Parallel(n_jobs=-1)(
        delayed(simulate_damped_pendulum)(sample, i) for i, sample in tqdm(enumerate(samples), total=len(samples))
    )

    # Store results in the correct order
    for index, solution in simulations:
        results[index, :] = solution

    return results


# Generate and save datasets
train_dataset = create_dataset("data/train_samples.npy")
test_dataset = create_dataset("data/test_samples.npy")

np.save("data/train_dataset.npy", train_dataset)
np.save("data/test_dataset.npy", test_dataset)
