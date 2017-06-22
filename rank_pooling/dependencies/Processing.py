import os
import numpy as np

def sorted_frames_by_substring(dir_name, substring_name):
    folder = [f for f in os.listdir(dir_name) if substring_name in f]
    frame_number_array = sorted(map(int, [f.split('_')[0] for f in folder]))
    return [str(s) + substring_name for s in frame_number_array]


def calculate_sse(a, b):
    return np.sum(np.square(a-b))


def calculate_magnitude(data): #this will give magnitude of every frame with 9 values
    return np.linalg.norm(data)


def get_magnitude(data_list): #this will give magnitude of every window of (window_size, 9)
    if data_list == []:
        return [0]
    else:
        magnitude = []
        mag = 0
        #for i in range(len(data_list)):
        for j in range(data_list.shape[0] - 1):
            mag += calculate_magnitude(data_list[j + 1] - data_list[j])
        magnitude.append(mag)
        return magnitude


def divide_list(combination_list, index, machine_count):
    unit = len(combination_list) / machine_count
    b = combination_list[(index * unit): ((index + 1) * unit)]
    #b = combination_list[(index*unit):((index+1)*unit)]
    return b