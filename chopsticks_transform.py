import json


# load original data
demo_path = "chopsticks/demo.txt"
with open(demo_path, "r") as file:
    motion_dict = json.load(file)


# load inbetweened data
inbetweened_file_name = "missing=0.5_method=MiceImputation_trial=0_time=1.6340434551239014.txt"
inbetweened_path = "chopsticks/data/" + inbetweened_file_name
inbetweened_data = []
with open(inbetweened_path, "r") as file:
    for line in file:
        frame = [float(value) for value in line.strip().split(",")]
        inbetweened_data.append(frame)


# transform inbetweened data back to the original format of motion data
graphic_params_keys = ['openloop_motion', 'vel_openloop', 'openloop_arm', 'vel_openloop_arm', 'motion_object', 'motion_chopsticks', 'vel_chopsticks']
index = 1   # column index
for key in graphic_params_keys:
    start = index   # inclusive
    end = index + len(motion_dict[key][0])  # exclusive
    motion_dict[key] = [list(frame[start:end]) for frame in inbetweened_data]
    index = end


# save inbetweened data to a file
file_name = f"chopsticks/0.5_mice.txt"
with open(file_name, "w") as file:
    json.dump(motion_dict, file)
