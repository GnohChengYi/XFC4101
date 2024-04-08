from math import cos, sin
from motion import *
import numpy as np
import random
import time
from visualization import plot
from Method.knn import KNNImputation
from Method.missforest import MissForestImputation
from Method.mice import MiceImputation
from Method.lerp import LinearInterpolator
from Method.slerp import SphericalLinearInterpolator


DEBUG = False

# initialize full data
from_t = 0
to_t = 10 if DEBUG else 1000

position0 = [0, 0]
velocity0 = [1, 2]
acceleration = [0.3, 0.4]
motion = UniformlyAcceleratedMotion(position0, velocity0, acceleration)
# x_radius = 2
# y_radius = 3
# motion = EllipticalMotion(x_radius, y_radius)
# amplitude = 2
# frequency = 0.1
# phase = 3
# fixed_axis_values = [4, 5]
# motion = SimpleHarmonicMotion(amplitude, frequency, phase, fixed_axis_values)


full_data = motion.get_frames(from_t, to_t)
if DEBUG:
    print(full_data)
else:
    file_name = f"Data/truth_{str(motion)}.txt"
    with open(file_name, "w") as file:
        for frame in full_data:
            file.write(",".join(str(value) for value in frame) + "\n")
    print("Full data saved to:", file_name)



random.seed(4101)

num_trials = 2 if DEBUG else 10

missing_fractions = [0.5, 0.6, 0.7, 0.8, 0.9]
missing_fraction = 0.5      ################################################ fix missing_fraction for each run
print(f'missing_fraction: {missing_fraction}')

methods = [KNNImputation(), MissForestImputation(), MiceImputation(), LinearInterpolator(), SphericalLinearInterpolator()]
if DEBUG:
    pass


ranks = [[] for _ in range(len(methods))]   # each [] will store the ranks (for each trial) of the method in the corresponding index of methods
for trial in range(num_trials):
    print(f'\ntrial: {trial+1} / {num_trials}')
    
    # prepare keyframes by removing a fraction of data
    num_rows = len(full_data)
    num_missing_rows = int((num_rows - 2) * missing_fraction)
    missing_rows = random.sample(range(1, num_rows - 1), num_missing_rows)
    keyframes = [frame.copy() for frame in full_data]
    for row in missing_rows:
        keyframes[row][1:] = [None] * (len(keyframes[row]) - 1)
    if DEBUG:
        print(keyframes)
    
    RMSEs = []  # root mean squared errors for each method
    for method in methods:
        print(f'method: {method.__class__.__name__}')
        
        # generate inbetweened data        
        start_time = time.time()
        inbetweened_data = method.inbetween(keyframes)
        end_time = time.time()
        execution_time = end_time - start_time
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
        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(inbetweened_data, full_data, squared=False)
        RMSEs.append(rmse)
        if DEBUG:
            print("RMSE:", rmse)
    
    # rank the methods based on RMSE
    ranked_methods = sorted(range(len(methods)), key=lambda i: RMSEs[i])
    print()
    for i, method in enumerate(methods):
        ranks[i].append(ranked_methods.index(i) + 1)
        print(f'{method.__class__.__name__}: {RMSEs[i]}')
        print(f'rank: {ranked_methods.index(i) + 1}')



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
