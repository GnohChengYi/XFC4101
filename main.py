from motion import Motion
import random
import time
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
position_function = lambda t: [position0[0] + velocity0[0] * t + 0.5 * acceleration[0] * t**2, position0[1] + velocity0[1] * t + 0.5 * acceleration[1] * t**2]
motion = Motion(position_function)
full_data = motion.get_frames(from_t, to_t)
if DEBUG:
    print(full_data)
else:
    file_name = f"Data/truth_{position0}_{velocity0}_{acceleration}.txt"
    with open(file_name, "w") as file:
        for frame in full_data:
            file.write(",".join(str(value) for value in frame) + "\n")
    print("Full data saved to:", file_name)


random.seed(4101)
missing_fractions = [0.5, 0.6, 0.7, 0.8, 0.9]

for missing_fraction in missing_fractions:
    # prepare keyframes by removing a fraction of data
    num_rows = len(full_data)
    num_missing_rows = int((num_rows - 2) * missing_fraction)

    missing_rows = random.sample(range(1, num_rows - 1), num_missing_rows)
    keyframes = [frame.copy() for frame in full_data]
    for row in missing_rows:
        keyframes[row][1:] = [None] * (len(keyframes[row]) - 1)
    if DEBUG:
        print(f'missing_fraction: {missing_fraction}')
        print(keyframes)
    

    # generate inbetweened data
    # method = KNNImputation()
    # method = MissForestImputation()
    # method = MiceImputation()
    # method = LinearInterpolator()
    method = SphericalLinearInterpolator()
    start_time = time.time()
    inbetweened_data = method.inbetween(keyframes)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    if DEBUG:
        print(inbetweened_data)


    # save inbetweened_data to a file
    if not DEBUG:
        file_name = f"Data/inbetweened_{method.__class__.__name__}_{missing_fraction}_{position0}_{velocity0}_{acceleration}_{execution_time}.txt"
        with open(file_name, "w") as file:
            for frame in inbetweened_data:
                file.write(",".join(str(value) for value in frame) + "\n")
        print("Inbetweened data saved to:", file_name)


    # compare inbetweened data with original data
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(inbetweened_data, full_data, squared=False)
    print("RMSE:", rmse)
