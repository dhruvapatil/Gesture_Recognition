import os
import skimage.io
from skimage.transform import resize, rescale, pyramid_reduce
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


def get_palm_points(skeleton_file):
    palm_dict = {}
    data = open(skeleton_file).readlines()[1:]
    for line in data:
        line_list = line.split(", ")

        frame = line_list[0]
        try:
            hand_left = map(float, line_list[64:66])
            hand_right = map(float, line_list[96:98])

            palm_dict[frame] = [hand_left, hand_right]
        except:
            pass

    return palm_dict

def get_heat_map(index_array, left, right):

    left_array = np.zeros(index_array.shape)

    if left[0] is None:
        heat_map_0 = left_array[:,:,0]
    else:
        left_array[:, :, 0] = left[1]
        left_array[:, :, 1] = left[0]
        heat_map_0 = np.exp(-np.sqrt(np.sum(np.square(index_array - left_array), axis=2)))

    right_array = np.zeros(index_array.shape)

    if right[0] is None:
        heat_map_1 = right_array[:,:,0]
    else:
        right_array[:, :, 0] = right[1]
        right_array[:, :, 1] = right[0]
        heat_map_1 = np.exp(-np.sqrt(np.sum(np.square(index_array - right_array), axis=2)))

    heat_map = np.dstack([heat_map_0, heat_map_1])
    return heat_map

def save_dataset():
    labels_root = "/s/red/a/nobackup/cwc/palm_recognition/palm_labels/"
    labels = np.vstack([np.load(os.path.join(labels_root,numpy_file)) for numpy_file in os.listdir(labels_root)])


    index_array = np.zeros((240/8, 320/8, 2))

    for i in range(index_array.shape[0]):
        for j in range(index_array.shape[1]):
            index_array[i][j] = [i, j]

    X = []
    y = []
    for file_name, l_x, l_y, r_x, r_y in labels:

        left = [l/8  if l is not None else l for l in [l_x, l_y]]
        right = [r/8 if r is not None else r for r in [r_x, r_y]]

        print file_name, left, right

        heat_map = get_heat_map(index_array, left, right)

        image = skimage.io.imread(file_name, True)

        X.append(image)
        y.append(heat_map)

        '''image = rescale(image, 1/8.0)

        plt.subplot(121)
        plt.imshow(image,cmap="gray")
        plt.imshow(heat_map[:, :, 0],alpha=0.5)

        plt.subplot(122)
        plt.imshow(image,cmap="gray")
        plt.imshow(heat_map[:, :, 1],alpha=0.5)
        plt.show()'''


    X = np.stack(X)
    y = np.stack(y)


    X = X[:,:,:,np.newaxis]

    indices = range(X.shape[0])
    shuffle(indices)
    print indices

    X_train = X[indices[:8000],:,:,:]
    y_train = y[indices[:8000], :, :, :]
    X_validation = X[indices[8000:], :, :, :]
    y_validation = y[indices[8000:], :, :, :]

    print X.shape, y.shape, np.min(X), np.max(X), np.min(y), np.max(y)
    print X_train.shape, y_train.shape, X_validation.shape, y_validation.shape

    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/X_train.npy", X_train)
    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/y_train.npy", y_train)

    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/X_validation.npy", X_validation)
    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/y_validation.npy", y_validation)

    return

def get_mean_image():
    sum = np.zeros((240,320))
    count  = 0
    for i in range(1,35879):
        folder = ((i - 1) / 200) + 1
        root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/train/%03d/K_%05d"%(folder,i)
        print folder, i

        frame_list = os.listdir(root)
        shuffle(frame_list)
        for f in frame_list[:5]:
            sum += skimage.io.imread(os.path.join(root, f),"True")
            count += 1

    mean = sum/count
    np.save("/s/red/a/nobackup/cwc/palm_recognition/dataset/mean.npy",mean)
    print np.max(mean), np.min(mean)
    plt.imshow(mean,cmap="gray")
    plt.show()


if __name__ == "__main__":
    #save_dataset("Depth")
    save_dataset()
    #get_mean_image()