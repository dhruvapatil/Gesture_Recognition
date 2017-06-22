import numpy as np
import math

def calculate_distance_between_wrists(data):
    #data format: [spine, right elbow, right wrist, left elbow, left wrist]
    #index format: [0,1,   2,3,           4,5,     6,7,          8,9]
    wrist_right = data[:, [4,5]]
    wrist_left = data[:, [8,9]]

    #print wrist_left[[7,8,9,10], :]
    #print wrist_right[[7,8,9,10], :]
    #print len(wrist_right), len(wrist_left)
    frames = len(wrist_right)
    dist = np.zeros(frames)
    for i in range(frames):
        dist[i] = math.hypot(wrist_right[i,0] - wrist_left[i,0], wrist_right[i,1] - wrist_right[i,1])

    dist = dist[1:-1]
    return np.array(dist)


def calculate_distance_from_spine(normalized_video):
    spine = normalized_video[:, [0,1]]
    wrist = normalized_video[:, [4,5]]
    frames = len(wrist)
    dist = np.zeros(frames)

    for i in range(frames):
        dist[i] = math.hypot(wrist[i,0] - spine[i,0], wrist[i,1] - spine[i,1])
    dist = dist[1:-1]
    return np.array(dist)

def calculate_product_of_joints(data):
    frames = data.shape[0]
    product = np.zeros(frames)

    for i in range(frames):
        for j in range(4):
            product[i] = data[i][j] * data[i][j+4]
    return product


'''
filename = '../skeleton/vid.npy'

a = np.load(filename)
print a.shape
distance_of_wrists = calculate_distance_between_wrists(a)
distance_spine_wrist = calculate_distance_from_spine(a)

print distance_of_wrists.shape
print distance_spine_wrist.shape
'''
