import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes



def encode_label(label):
    return int(label)


def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
	if len(line) == 1:
	    continue
        filepath, label = line.split(" ")
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels


def get_input(batch_size, hand):

    train_labels_file = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/train_frame_%s_list.txt"%hand
    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 100
    NUM_CHANNELS = 1
    BATCH_SIZE = batch_size

    # reading labels and file path
    train_filepaths, train_labels = read_label_file(train_labels_file)
    print len(train_filepaths), len(train_labels)

    # convert string into tensors
    train_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
    train_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.int32)



    # create input queues
    train_input_queue = tf.train.slice_input_producer(
        [train_images, train_labels],
        shuffle=True)


    # process path and string tensor into an image and a label
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
    train_label = tf.one_hot(tf.sub(train_input_queue[1],1),depth=250)

    train_image = tf.image.convert_image_dtype(train_image,dtypes.float32)
    train_image = tf.sub(train_image, 0.5)
    train_image = tf.mul(train_image, 2.0)



    # define tensor shape
    train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    # collect batches of images before processing
    train_image_batch, train_label_batch = tf.train.batch([train_image, train_label],batch_size=BATCH_SIZE,num_threads=16)


    print "input pipeline ready"

    return train_image_batch, train_label_batch

if __name__ == "__main__":
    train_image_batch, train_label_batch = get_input(10,"right")

    with tf.Session() as sess:
        # initialize the variables
        #sess.run(tf.initialize_all_variables())

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print "from the train set:"
        for i in range(20):
            images, labels = sess.run([train_image_batch,train_label_batch])
            print images.shape, labels.shape, np.max(images), np.min(images)

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()
