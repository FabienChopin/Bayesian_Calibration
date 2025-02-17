import warnings

import numpy as np
import tensorflow as tf
from keras import Model, Input, layers, backend as K

from custom_callbacks import CustomCheckpoint


def build_model():
    """
    Builds and compiles a feedforward neural network model.

    Returns:
    - Compiled Keras model.
    """
    K.clear_session()

    input_layer = Input(shape=(7,))
    x = layers.Dense(64, activation="relu")(input_layer)
    for _ in range(4):
        x = layers.Dense(64, activation="relu")(x)
    output_layer = layers.Dense(1, activation="linear")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse")

    return model


def create_tf_dataset(
        param_file: str,
        solutions_file: str,
        batch_size: int = 1024,
        cache: bool = True,
        prefetch: bool = True
) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from parameter and solution files.

    Args:
    - param_file (str): Path to the file containing input parameters.
    - solutions_file (str): Path to the file containing solution data (time and angles).
    - batch_size (int, optional): Batch size for training. Default = 1024.
    - cache (bool, optional): Enables dataset caching if memory allows. Default = True.
    - prefetch (bool, optional): Enables prefetching for improved training performance. Default = True.

    Returns:
    - tf.data.Dataset: TensorFlow dataset ready for model training.
    """
    parameters = np.load(param_file)
    solutions = np.load(solutions_file)

    # Validate dataset dimensions
    if parameters.shape[0] != solutions.shape[0]:
        raise ValueError(f"Mismatch between parameters ({parameters.shape}) and solutions ({solutions.shape})")

    if solutions.shape[1] % 2 != 0:
        raise ValueError(
            f"Solution column count ({solutions.shape[1]}) must be even to separate time and angles correctly.")

    # Split time and angle data
    split_idx = solutions.shape[1] // 2
    times, angles = solutions[:, :split_idx], solutions[:, split_idx:]
    num_timesteps = times.shape[1]

    # Flatten time series data and repeat parameters accordingly
    times_flat = times.flatten()
    angles_flat = angles.flatten()
    parameters_repeated = np.repeat(parameters, num_timesteps, axis=0)

    # Construct final dataset
    dataset_data = np.column_stack((parameters_repeated, times_flat, angles_flat))
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset_data[:, :-1], dataset_data[:, -1:])
    ).batch(batch_size)

    # Apply optimizations
    if cache and dataset_data.nbytes <= 1e9:
        tf_dataset = tf_dataset.cache()
    elif cache:
        warnings.warn(f"Dataset too large to cache ({dataset_data.nbytes / 1e9:.2f} GB > 1 GB).", category=UserWarning)

    if prefetch:
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


# Load datasets
train_tfds = create_tf_dataset("data/train_samples.npy", "data/train_dataset.npy", batch_size=4096)
test_tfds = create_tf_dataset("data/test_samples.npy", "data/test_dataset.npy", batch_size=4096)

# Train model
model = build_model()
checkpoint_callback = CustomCheckpoint(train_tfds, test_tfds, "models_saved/best_model.keras")
model.fit(train_tfds, epochs=100, callbacks=[checkpoint_callback])
