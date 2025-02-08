import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Nombre d'échantillons aléatoires
number_of_samples = 100000

param_matrix = np.random.uniform(low=np.array([[9.8, 0.5, 0.5, 0.8, -0.5*np.pi, -0.01]]),
                                 high=np.array([[9.82, 1.5, 1.5, 1.2, 0.5*np.pi, 0.01]]),
                                 size=(number_of_samples, 6))

np.save("uniform_samples.npy", param_matrix)

# Affichage des histogrammes dans une future version ?


