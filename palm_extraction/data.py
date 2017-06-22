import numpy as np
import os
import base
from random import choice
from skimage.transform import resize
import matplotlib.pyplot as plt

class DataSet(object):

    def __init__(self, data, labels):
        self._num_examples = data.shape[0]
        self._indices = np.arange(self._num_examples)
        mean_image = np.load("/s/red/a/nobackup/cwc/palm_recognition/dataset/mean.npy")[:,:,np.newaxis]
        #data = data.astype(np.float32)
        print np.min(data), np.max(data), np.min(mean_image), np.max(mean_image)
        data-=mean_image

        print np.min(data), np.max(data)
        self._images = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        random_indices = [choice(self._indices) for _ in range(batch_size)]
        return self._images[random_indices,:,:,:], self._labels[random_indices,:,:,:]

def read_datasets():
    data_dir = "/s/red/a/nobackup/cwc/palm_recognition/dataset/"
    print "Loading data ", data_dir

    X_train = np.load(os.path.join(data_dir,"X_train.npy"))
    y_train = np.load(os.path.join(data_dir,"y_train.npy"))

    X_validation = np.load(os.path.join(data_dir, "X_validation.npy"))
    y_validation = np.load(os.path.join(data_dir, "y_validation.npy"))

    print X_train.shape, y_train.shape, X_validation.shape, y_validation.shape

    train = DataSet(X_train, y_train)
    validation = DataSet(X_validation, y_validation)
    print "Loading done"
    return base.Datasets(train=train, validation=validation, test = None)

def load_data():
    return read_datasets()

if __name__ == "__main__":
    data = load_data()
    x, y = data.train.next_batch(10)
    mean_image = np.load("/s/red/a/nobackup/cwc/palm_recognition/dataset/mean.npy")[:,:,np.newaxis]

    print x.shape, y.shape

    for image, heat_map in zip(x,y):
        print image.shape

        image += mean_image

        image = resize(image,(heat_map.shape[0], heat_map.shape[1],image.shape[-1]),preserve_range=True)

        image = image[:,:,0]

        print np.min(image), np.max(image), image.shape,heat_map.shape
        plt.subplot(221)
        plt.imshow(image)
        plt.imshow(heat_map[:, :, 0], alpha=0.5)

        plt.subplot(222)
        plt.imshow(image)
        plt.imshow(heat_map[:, :, 1], alpha=0.5)

        plt.subplot(223)
        plt.imshow(image)

        plt.subplot(224)
        plt.imshow(image)

        plt.show()


