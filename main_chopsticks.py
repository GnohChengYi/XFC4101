import json
import random
import requests
from sklearn.metrics import mean_squared_error
import time
from Method.knn import KNNImputation
from Method.lerp import LinearInterpolator
from Method.mice import MiceImputation
from Method.missforest import MissForestImputation
from Method.slerp import SphericalLinearInterpolator
import os


DEBUG = True


# load demo data
# use demo data from https://github.com/chopsticks-research2022/learning2usechopsticks to conduct experiments
demo_path = "chopsticks/demo.txt"
if not os.path.exists(demo_path):
    url = "https://raw.githubusercontent.com/chopsticks-research2022/learning2usechopsticks/main/released_models/demo.txt"
    response = requests.get(url)
    if response.status_code == 200:
        with open(demo_path, "wb") as file:
            file.write(response.content)
            print("File downloaded successfully.")
    else:
        print("Failed to download the file.")

with open(demo_path, "r") as file:
    motion_dict = json.load(file)


# transform data into a suitable format for inbetweening
graphic_params_keys = ['openloop_motion', 'vel_openloop', 'openloop_arm', 'vel_openloop_arm', 'motion_object', 'motion_chopsticks', 'vel_chopsticks']

from_t = 0
to_t = len(motion_dict[graphic_params_keys[0]]) - 1

full_data = []
for t in range(from_t, to_t + 1):
    frame = [t]
    for key in graphic_params_keys:
        frame.extend(motion_dict[key][t])
    full_data.append(frame)


# fix experiment parameters
random.seed(4101)
num_trials = 2
methods = [KNNImputation(), LinearInterpolator()]
missing_fractions = [0.5, 0.6]


# conduct experiments
missing_RMSEs = [[] for _ in range(len(methods))]  # [method][missing_fraction][RMSE]
missing_ranks = [[] for _ in range(len(methods))] # [method][missing_fraction][rank]
missing_times = [[] for _ in range(len(methods))] # [method][missing_fraction][time]
for missing_fraction in missing_fractions:
    print(f"Missing fraction: {missing_fraction}")
    RMSEs = [[] for _ in range(len(methods))]   # each [] will store the RMSEs (for each trial) of the method in the corresponding index of methods
    ranks = [[] for _ in range(len(methods))]   # each [] will store the ranks (for each trial) of the method in the corresponding index of methods
    times = [[] for _ in range(len(methods))]   # each [] will store the times (for each trial) of the method in the corresponding index of methods
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1} / {num_trials}")
        
        # prepare keyframes by removing a fraction of data
        num_rows = len(full_data)
        num_missing_rows = int((num_rows - 2) * missing_fraction)
        missing_rows = random.sample(range(1, num_rows - 1), num_missing_rows)
        keyframes = [frame.copy() for frame in full_data]
        for row in missing_rows:
            keyframes[row][1:] = [None] * (len(keyframes[row]) - 1)

        method_RMSE = []  # root mean squared errors for each method
        method_time = [] # execution times for each method
        for method in methods:
            print(f"Method: {method.__class__.__name__}")
            
            # generate inbetweened data
            start_time = time.time()
            inbetweened_data = method.inbetween(keyframes)
            end_time = time.time()
            execution_time = end_time - start_time
            method_time.append(execution_time)
            
            # compute quality
            rmse = mean_squared_error(inbetweened_data, full_data, squared=False)
            method_RMSE.append(rmse)
            if DEBUG:   print(f"RMSE: {rmse}")

        # rank the methods based on RMSE
        ranked_methods = sorted(range(len(methods)), key=lambda i: method_RMSE[i])
        print()
        for i, method in enumerate(methods):
            RMSEs[i].append(method_RMSE[i])
            ranks[i].append(ranked_methods.index(i) + 1)
            times[i].append(method_time[i])
            print(f"{method.__class__.__name__}: RMSE={method_RMSE[i]}")
            print(f"{method.__class__.__name__}: Rank={ranks[i][-1]}")

    for i, method in enumerate(methods):
        average_RMSE = sum(RMSEs[i]) / num_trials
        missing_RMSEs[i].append(average_RMSE)
        average_rank = sum(ranks[i]) / num_trials
        missing_ranks[i].append(average_rank)
        average_time = sum(times[i]) / num_trials
        missing_times[i].append(average_time)


print(f'Missing RMSEs: {missing_RMSEs}')
print(f'Missing ranks: {missing_ranks}')
print(f'Missing times: {missing_times}')

'''
# transform inbetweened data back to the original format of motion data
inbetweened_motion_dict = motion_dict.copy()
index = 1   # column index
for key in graphic_params_keys:
    start = index   # inclusive
    end = index + len(motion_dict[key][0])  # exclusive
    inbetweened_motion_dict[key] = [frame[start:end] for frame in inbetweened_data]
    index = end


# save inbetweened data to a file
file_name = f"chopsticks/inbetweened_{missing_fraction}.txt"
with open(file_name, "w") as file:
    json.dump(inbetweened_motion_dict, file)
'''