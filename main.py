from math import cos, sin
from motion import *
import numpy as np
import random
import time
from visualization import *
from Method.knn import KNNImputation
from Method.missforest import MissForestImputation
from Method.mice import MiceImputation
from Method.lerp import LinearInterpolator
from Method.slerp import SphericalLinearInterpolator
from sklearn.metrics import mean_squared_error


DEBUG = True

# initialize full data
from_t = 0
to_t = 10 if DEBUG else 1000

position0 = [0, 0]
velocity0 = [10, 10]
acceleration = [0, -0.02]
motion = UniformlyAcceleratedMotion(position0, velocity0, acceleration) # projectile motion
# position0 = [0, 0]
# velocity0 = [1, 2]
# acceleration = [0.3, 0.4]
# motion = UniformlyAcceleratedMotion(position0, velocity0, acceleration)
# x_radius = 20
# y_radius = 30
# motion = EllipticalMotion(x_radius, y_radius)
# radius = 2
# motion = CircularMotion(radius)
# amplitude = 2
# frequency = 0.1
# phase = 3
# fixed_axis_values = [4, 5]
# motion = SimpleHarmonicMotion(amplitude, frequency, phase, fixed_axis_values)


full_data = motion.get_frames(from_t, to_t)
if DEBUG:
    full_data = [[row[0], row[2]] for row in full_data]
    print(full_data)
    print(len(full_data))
else:
    file_name = f"Data/truth_{str(motion)}.txt"
    with open(file_name, "w") as file:
        for frame in full_data:
            file.write(",".join(str(value) for value in frame) + "\n")
    print("Full data saved to:", file_name)


random.seed(4101)

num_trials = 100 if DEBUG else 10

methods = [KNNImputation(), MissForestImputation(), MiceImputation(), LinearInterpolator(), SphericalLinearInterpolator()]
if DEBUG:
    methods = [LinearInterpolator()]
    pass

missing_fractions = [0.9] if DEBUG else [0.5, 0.6, 0.7, 0.8, 0.9]


missing_RMSEs = [[] for _ in range(len(methods))] # [method][missing_fraction][RMSE]
missing_ranks = [[] for _ in range(len(methods))] # [method][missing_fraction][rank]
missing_times = [[] for _ in range(len(methods))] # [method][missing_fraction][time]
for missing_fraction in missing_fractions:
    print(f'missing_fraction: {missing_fraction}')
    RMSEs = [[] for _ in range(len(methods))]   # each [] will store the RMSEs (for each trial) of the method in the corresponding index of methods
    ranks = [[] for _ in range(len(methods))]   # each [] will store the ranks (for each trial) of the method in the corresponding index of methods
    times = [[] for _ in range(len(methods))]   # each [] will store the times (for each trial) of the method in the corresponding index of methods
    for trial in range(num_trials):
        print(f'\ntrial: {trial+1} / {num_trials}')
        
        # prepare keyframes by removing a fraction of data
        num_rows = len(full_data)
        num_missing_rows = int((num_rows - 2) * missing_fraction)
        missing_rows = random.sample(range(1, num_rows - 1), num_missing_rows)
        keyframes = [frame.copy() for frame in full_data]
        for row in missing_rows:
            keyframes[row][1:] = [None] * (len(keyframes[row]) - 1)
        
        # plot_trajectory_from_data(keyframes, "Greys", "Keyframes")
        
        if DEBUG:
            print(keyframes)
        
        method_RMSE = []  # root mean squared errors for each method
        method_time = []  # execution times for each method
        for method in methods:
            print(f'method: {method.__class__.__name__}')
            
            # generate inbetweened data        
            start_time = time.time()
            inbetweened_data = method.inbetween(keyframes)
            end_time = time.time()
            execution_time = end_time - start_time
            method_time.append(execution_time)
            if DEBUG:
                print("Execution time:", execution_time, "seconds")
                print(inbetweened_data)

            # save inbetweened_data to a file
            if not DEBUG:
                file_name = f"Data/inbetweened_{str(motion)}_{method.__class__.__name__}_{missing_fraction}_{execution_time}_{trial}.txt"
                with open(file_name, "w") as file:
                    for frame in inbetweened_data:
                        file.write(",".join(str(value) for value in frame) + "\n")
                print("Inbetweened data saved to:", file_name)

            # compare inbetweened data with original data
            rmse = mean_squared_error(inbetweened_data, full_data, squared=False)
            method_RMSE.append(rmse)
            if DEBUG:
                print("RMSE:", rmse)
        
        # rank the methods based on RMSE
        ranked_methods = sorted(range(len(methods)), key=lambda i: method_RMSE[i])
        print()
        for i, method in enumerate(methods):
            RMSEs[i].append(method_RMSE[i])
            ranks[i].append(ranked_methods.index(i) + 1)
            times[i].append(method_time[i])
            print(f'{method.__class__.__name__}: {method_RMSE[i]}')
            print(f'rank: {ranked_methods.index(i) + 1}')
    
    for i, method in enumerate(methods):
        average_RMSE = sum(RMSEs[i]) / num_trials
        missing_RMSEs[i].append(average_RMSE)
        average_rank = sum(ranks[i]) / num_trials
        missing_ranks[i].append(average_rank)
        average_time = sum(times[i]) / num_trials
        missing_times[i].append(average_time)


if DEBUG:
    print(missing_RMSEs)
    print(missing_ranks)
    print(missing_times)
    exit()


# '''
import matplotlib.pyplot as plt

method_labels = ["KNN", "MissForest", "MICE", "LERP", "SLERP"]

def plot_results(xlabel, ylabel, method_labels, DEBUG):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(method_labels)
    plt.title(f'{ylabel} vs {xlabel}')
    if not DEBUG:
        path = f'Results/{ylabel}.png'
        plt.savefig(path)
        print(f"Figure saved to {path}")
    plt.show()

xlabel = "Missing Fraction"

for i, method in enumerate(methods):
    plt.plot(missing_fractions, missing_RMSEs[i])
ylabel = "RMSE"
plot_results(xlabel, ylabel, method_labels, DEBUG)

for i, method in enumerate(methods):
    plt.plot(missing_fractions, missing_ranks[i])
ylabel = "Rank"
plot_results(xlabel, ylabel, method_labels, DEBUG)

for i, method in enumerate(methods):
    plt.plot(missing_fractions, missing_times[i])
ylabel = "Execution Time"
plot_results(xlabel, ylabel, method_labels, DEBUG)

exit()
# '''




# Compare and print results
print("\nMotion:", motion)
print("Missing Fraction:", missing_fraction)

for i, r in enumerate(ranks):
    minimum = np.min(r)
    maximum = np.max(r)
    median = np.median(r)
    first_quartile = np.percentile(r, 25)
    third_quartile = np.percentile(r, 75)
    
    print()
    print(str(methods[i]))
    print("Minimum:", minimum)
    print("Maximum:", maximum)
    print("Median:", median)
    print("First Quartile:", first_quartile)
    print("Third Quartile:", third_quartile)


# Save results to a file
file_name = f"Data/results_{motion}_{missing_fraction}.txt"
with open(file_name, "w") as file:
    file.write("Motion: " + str(motion) + "\n")
    file.write("Missing Fraction: " + str(missing_fraction) + "\n\n")
    
    for i, r in enumerate(ranks):
        file.write(str(methods[i]) + "\n")
        file.write(str(r) + "\n\n")
    
print("Results saved to:", file_name)


# Plot and save results
labels = ["KNN", "MissForest", "MICE", "LERP", "SLERP"]
ylabel = "Quality Rank"
title = f"{motion.__class__.__name__}, {int(missing_fraction * 100)}% missing data"
plot(ranks, labels, ylabel, title, save=not DEBUG)
