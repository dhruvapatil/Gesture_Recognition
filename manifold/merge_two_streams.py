import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def get_gt_labels():
    label_file = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/train_list.txt"
    label_dict = {}
    for line in open(label_file).readlines():
        _, path, label = line.strip().split(" ")
        path = path.split("_")[-1].replace(".avi", "")
        label_dict[int(path)] = int(label) - 1
    return label_dict


def random_sample_labels():
    label_dict = get_gt_labels()
    labels = np.array([label_dict[i] for i in range(1,30001)])
    subset = []
    for i in range(249):
        print i, np.ravel(np.argwhere(labels==i)).shape
        subset.append(np.random.choice(np.ravel(np.argwhere(labels==i)),50,False))


    subset = list(set(np.hstack(subset)))
    subset.sort()
    diff_set = list(set(range(0,30000)).difference(set(subset)))
    return subset, diff_set


def visualize_cm():
    d = np.load("cm_1_M.npy")
    a = np.load("cm_K.npy")
    c = np.load("cm_1_K.npy")
    b = np.load("cm_M.npy")

    M = b + d
    K = a + c

    M /= np.sum(M, axis=1)[:, np.newaxis]
    K /= np.sum(K, axis=1)[:, np.newaxis]

    plt.imshow(M, interpolation="nearest")
    plt.title("M")
    plt.show()

    plt.imshow(K, interpolation="nearest")
    plt.title("K")
    plt.show()

subset, diff_set = random_sample_labels()
label_dict = get_gt_labels()
root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/prediction/train/%03d/%s_%05d/distance.npy"

total_videos = 0.0
rgb_tp = 0.0
depth_tp = 0.0
combined_tp = 0.0
cm_combined = np.zeros((249,249))

for video in range(30001,31601):
    total_videos += 1
    folder = ((video - 1) / 200) + 1

    depth = np.load(root%(folder,"K",video))[:,0]
    rgb = np.load(root%(folder,"M",video))[:,0]


    #rgb[diff_set] = 0
    #depth[diff_set]= 0

    combined = rgb + depth


    gt = label_dict[video]

    #print rgb.shape, depth.shape

    rgb_sorted_indices = np.argsort(rgb)
    depth_sorted_indices = np.argsort(depth)
    combined_sorted_indices = np.argsort(combined)

    rgb_predictions = [label_dict[i+1] for i in rgb_sorted_indices][-5:]
    depth_predictions = [label_dict[i+1] for i in depth_sorted_indices][-5:]
    combined_predictions = [label_dict[i+1] for i in combined_sorted_indices][-5:]

    if rgb_predictions[-1]==gt:
        rgb_tp += 1
    if depth_predictions[-1] == gt:
        depth_tp += 1
    if combined_predictions[-1] == gt:
        combined_tp += 1
    cm_combined[gt][combined_predictions] += 1
    print zip(rgb_predictions, depth_predictions)
    print combined_predictions
    print video,label_dict[video]



print rgb_tp/total_videos, depth_tp/total_videos, combined_tp/total_videos

plt.imshow(cm_combined,interpolation="nearest")
plt.show()

cm_combined /= np.sum(cm_combined, axis=1)[:, np.newaxis]
plt.imshow(cm_combined,interpolation="nearest")
plt.show()