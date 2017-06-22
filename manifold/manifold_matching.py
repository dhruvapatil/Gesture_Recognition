import numpy as np
import os
from intra import distance
import sys
import matplotlib.pyplot as plt
from time import time


def read_tensors(dataset,file_list, stream):
    basis_list = []
    for index, f in enumerate(file_list):
        #if index%1000 == 0 and index>0:
        #    print index,
        folder, video, i = f.split("_")

        root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/svd_max_diff/%s/%03d/%s_%05d/" % (dataset,
            int(folder), stream, int(video))

        t1 = np.load(os.path.join(root, "%d_w.npy" % int(i)))
        t2 = np.load(os.path.join(root, "%d_h.npy" % int(i)))
        t3 = np.load(os.path.join(root, "%d_t.npy" % int(i)))

        basis_list.append([t1, t2, t3])
    print
    return basis_list


def get_cluster_representative(stream):
    distance_list = []
    cluster_center = []
    for gesture in range(249):
        file_list = np.load(
            "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/cluster_max_diff/%s/%d_files.npy" % (stream, gesture))
        distance_matrix = np.load(
            "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/cluster_max_diff/%s/%d_distance.npy" % (
            stream, gesture))

        distance_list.append(distance_matrix.reshape(-1))
        mean_distance = np.mean(distance_matrix, axis=0)
        index = np.argmax(mean_distance)
        print gesture, distance_matrix.shape, np.min(distance_matrix), np.max(distance_matrix), np.mean(
            distance_matrix), np.median(distance_matrix), index, file_list[index]
        cluster_center.append(file_list[index])

    basis_list = read_tensors(cluster_center)

    return basis_list


def get_gt_labels():
    label_file = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/train_list.txt"
    label_dict = {}
    for line in open(label_file).readlines():
        _, path, label = line.strip().split(" ")
        path = path.split("_")[-1].replace(".avi", "")
        label_dict[int(path)] = int(label) - 1
    return label_dict


def get_distance_matrix(tensor_list1, tensor_list2):
    print "Tensors ", len(tensor_list1), len(tensor_list2)
    distance_matrix = np.zeros((len(tensor_list1), len(tensor_list2)))
    for index1, t in enumerate(tensor_list1):
        # if index1%100 == 0:
        #    print index1,
        for index2, v in enumerate(tensor_list2):
            dist = distance(t, v)
            distance_matrix[index1, index2] = dist
    # print
    return distance_matrix


def cluster_distance(stream):
    cluster_centers = get_cluster_representative(stream)

    file_list = []
    label_dict = get_gt_labels()
    for video in range(30001, 35879):
        folder = ((video - 1) / 200) + 1
        file_list.append("%d_%d_%d" % (folder, video, 0))

    basis_list = read_tensors(file_list)

    distance_matrix = get_distance_matrix(cluster_centers, basis_list)
    print distance_matrix.shape
    print np.argmax(distance_matrix, axis=0)
    tp5 = 0.0
    fp5 = 0.0
    tp = 0.0
    fp = 0.0

    for i in range(distance_matrix.shape[1]):
        pred = list(np.argsort(distance_matrix[:, i])[-5:])
        print pred, label_dict[30001 + i], np.argwhere(np.argsort(distance_matrix[:, i]) == label_dict[30001 + i])
        if label_dict[30001 + i] in pred:
            tp5 += 1
        else:
            fp5 += 1

        if label_dict[30001 + i] == pred[-1]:
            tp += 1
        else:
            fp += 1
    print tp, fp, tp5, fp5
    print tp / (tp + fp), tp5 / (tp5 + fp5)


def video_distance_1(stream):
    label_dict = get_gt_labels()
    file_list = []
    for video in range(1, 30001):
        folder = ((video - 1) / 200) + 1
        file_list.append("%d_%d_%d" % (folder, video, 0))

    train_basis_list = read_tensors(file_list, stream)

    file_list = []
    for video in range(31001, 32001):
        folder = ((video - 1) / 200) + 1
        file_list.append("%d_%d_%d" % (folder, video, 0))

    test_basis_list = read_tensors(file_list, stream)

    distance_matrix = get_distance_matrix(train_basis_list, test_basis_list)
    print distance_matrix.shape
    print np.argmax(distance_matrix, axis=0)
    tp5 = 0.0
    fp5 = 0.0
    tp = 0.0
    fp = 0.0
    cm = np.zeros((249, 249))

    for i in range(distance_matrix.shape[1]):
        indices = np.argsort(distance_matrix[:, i])[-5:]
        pred = [label_dict[index + 1] for index in indices]
        if label_dict[31001 + i] in pred:
            tp5 += 1
        else:
            fp5 += 1

        if label_dict[31001 + i] == pred[-1]:
            tp += 1
        else:
            fp += 1
        cm[label_dict[31001 + i], pred[-1]] += 1
    print stream, tp, fp, tp5, fp5
    print stream, tp / (tp + fp), tp5 / (tp5 + fp5)
    np.save("cm_1_%s.npy" % stream, cm)





def video_distance(stream, start, end):
    file_list = []
    for video in range(1, 35879):
        folder = ((video - 1) / 200) + 1
        file_list.append("%d_%d_%d" % (folder, video, 0))

    train_basis_list = read_tensors("train",file_list, stream)

    for video in range(start, end):
        folder = ((video - 1) / 200) + 1
        dest_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/prediction/valid/%03d/%s_%05d" % (
        folder, stream, video)

        file_list = ["%d_%d_%d" % (folder, video, 0)]

        test_basis_list = read_tensors("valid",file_list, stream)

        distance_matrix = get_distance_matrix(train_basis_list, test_basis_list)
        print video, distance_matrix.shape

        try:
            os.makedirs(dest_root)
        except:
            pass
        np.save(os.path.join(dest_root, "distance.npy"), distance_matrix)


begin_time = time()
start = int(sys.argv[1])
end = start + 200
stream = "K"
video_distance(stream, start, end)
print "Total time ", time() - begin_time

