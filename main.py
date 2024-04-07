from motion import Motion
import random


# initialize full data
from_t = 0
to_t = 100

position0 = [0, 0]
velocity0 = [1, 2]
acceleration = [3, 4]
position_function = lambda t: [position0[0] + velocity0[0] * t + 0.5 * acceleration[0] * t**2, position0[1] + velocity0[1] * t + 0.5 * acceleration[1] * t**2]
motion = Motion(position_function)
full_data = motion.get_frames(from_t, to_t)
print(full_data)


# prepare keyframes by removing a fraction of data
missing_fraction = 0.2
num_rows = len(full_data)
num_missing_rows = int((num_rows - 2) * missing_fraction)

random.seed(4101)
missing_rows = random.sample(range(1, num_rows - 1), num_missing_rows)
keyframes = [frame.copy() for frame in full_data]
for row in missing_rows:
    keyframes[row][1:] = [None] * (len(keyframes[row]) - 1)
print(keyframes)


# generate inbetweened data

from Imputation.knn import KNNImputation
n_neighbors = 5
method = KNNImputation(n_neighbors=n_neighbors)
inbetweened_data = method.inbetween(keyframes)
print(inbetweened_data)


# save inbetweened_data to a file

file_name = f"Data/inbetweened_data_{method.__class__.__name__}_{missing_fraction}.txt"
with open(file_name, "w") as file:
    for frame in inbetweened_data:
        file.write(",".join(str(value) for value in frame) + "\n")
print("Inbetweened data saved to:", file_name)


# compare inbetweened data with original data
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(inbetweened_data, full_data, squared=False)
print("RMSE:", rmse)
