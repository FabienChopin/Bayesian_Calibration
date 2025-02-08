import numpy as np
from sklearn.model_selection import train_test_split
import scipy
import json

samples = np.load("uniform_samples.npy")
number_of_trials = 500
results = np.empty((number_of_trials, 2))
best_stat_mean = 1. # Worst statistic is 1 for each parameter
number_of_parameters = samples.shape[1]
best_train, best_test = None, None

for t in range(number_of_trials):
    train, test = train_test_split(samples, test_size=0.5)
    stat = []
    pvalue = []
    for i in range(number_of_parameters):
        ksresult = scipy.stats.kstest(train[:, i], test[:, i], alternative="two-sided")
        stat.append(ksresult.statistic)
        pvalue.append(ksresult.pvalue)

    mean_stat = np.mean(stat)
    if (mean_stat < best_stat_mean) and (np.min(pvalue) > 0.05):
        best_stat_mean = mean_stat
        best_train, best_test = train, test

if best_train is not None and best_test is not None:
    np.save("train_dataset.npy", best_train)
    np.save("test_dataset.npy", best_test)
    metadata = {"best_ks_stat_mean": best_stat_mean}
    with open("train_test_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Train test split saved for KS statitic of  {best_stat_mean:.4f}")
else:
    print("No valid split found, please consider increasing number_of_trials")

# We could load metadata at the beginning to be able to add new trials without erasing best split