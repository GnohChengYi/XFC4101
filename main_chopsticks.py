import json
import random
import requests
from sklearn.metrics import mean_squared_error
from Method.lerp import LinearInterpolator

# use demo data from https://github.com/chopsticks-research2022/learning2usechopsticks to conduct experiments

# load demo data
url = "https://raw.githubusercontent.com/chopsticks-research2022/learning2usechopsticks/main/released_models/demo.txt"
response = requests.get(url)

demo_path = "chopsticks/demo.txt"
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


# prepare keyframes by removing a fraction of data
missing_fraction = 0.5
num_rows = len(full_data)
num_missing_rows = int((num_rows - 2) * missing_fraction)
missing_rows = random.sample(range(1, num_rows - 1), num_missing_rows)
keyframes = [frame.copy() for frame in full_data]
for row in missing_rows:
    keyframes[row][1:] = [None] * (len(keyframes[row]) - 1)


# generate inbetweened data
method = LinearInterpolator()
inbetweened_data = method.inbetween(keyframes)


# compute quality
rmse = mean_squared_error(inbetweened_data, full_data, squared=False)
print(rmse)


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
