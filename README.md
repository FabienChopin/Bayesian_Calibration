# Pendulum Simulation and Neural Network Training

This project simulates the motion of a pendulum, including damped versions, using numerical methods and trains a neural network to predict the motion based on various input parameters.

## Overview

This project involves the following concepts:

1. **Pendulum Simulation**: 
   - We simulate the motion of a simple pendulum and a damped pendulum using the differential equations governing their movement. The `BasePendulum` class provides a base for these simulations, and `Pendulum` and `DampedPendulum` are specialized classes for the specific types of pendulums.
   - The simulation data is generated for different combinations of parameters such as gravitational acceleration, pendulum length, mass, and damping coefficient.

2. **Data Generation**:
   - The `generate_prior_samples.py` script generates a parameters' dataset according to a specified prior. This dataset is saved as uniform_samples.npy in the data/ folder. 
   - The `generate_datasets.py` script simulates pendulum motion for a set of parameters and generates time-series data for training. It saves the training and test datasets (`train_dataset.npy`, `test_dataset.npy`) along with their parameters (`train_samples.npy`, `test_samples.npy`) in the `data/` folder.

3. **Neural Network Training**:
   - A feedforward neural network is trained on the generated data to predict the pendulum’s motion based on input parameters. The training script (`train_nn.py`) utilizes TensorFlow/Keras for model training. A custom Keras callback (`callbacks.py`) monitors the model's performance and saves the best model during training.

## Requirements
You can install the required dependencies using:
```
pip install -r requirements.txt
```

## How to Run
### Generate parameters samples (prior)  
To generate the parameters according to the prior, run the following command:  
```
python generate_prior_samples.py
```

### Generate simulation data  

To generate the pendulum motion data, run the following command:  
```
python generate_datasets.py
```


### Train the Neural Network  
To train the neural network using the generated data, run:  
```
python train_nn.py
```  
The training will run for a set number of epochs and will save the best-performing model (based on a custom metric) as `best_model.keras` in the `models_saved/` folder. 

### Modify Parameters  
The simulation and neural network training can be customized by modifying the parameters in the following files:

   * `pendulum.py`: Customize the pendulum models (e.g., add new types of pendulums).
   * `generate_prior_samples.py`: Modify the parameters used for the prior.  
   * `train_nn.py`: Change the model architecture or training settings.
   * `custom_callbacks.py`: Change the custom metric used to save the model.