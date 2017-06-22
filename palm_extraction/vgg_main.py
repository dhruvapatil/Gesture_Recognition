import time
import numpy as np
import vgg_model
import tensorflow as tf
import os
from data import load_data
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage.io
import random
from scipy.ndimage.filters import gaussian_filter

os.environ["CUDA_VISIBLE_DEVICES"]="0"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_dir', '','Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '','Directory to keep eval outputs.')
tf.app.flags.DEFINE_string('log_root', '','Directory to keep the checkpoints. Should be a ''parent directory of FLAGS.train_dir/eval_dir.')



def train(hps):
    """Training loop."""

    model = vgg_model.VGG(hps, FLAGS.mode)
    model.build_graph()

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)
    saver = tf.train.Saver(max_to_keep=0)

    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=600,
                             saver=saver,
                             global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session()

    summary_writer.add_graph(sess.graph)

    step = 0
    lrn_rate = 1e-4

    dataset = load_data().train

    while not sv.should_stop():
        data, labels = dataset.next_batch(32)

        print step,
        (_, summaries, loss, predictions, truth, train_step) = sess.run(
            [model.train_op, model.summaries, model.cost, model.predictions,
             model.labels, model.global_step],
            feed_dict={model.lrn_rate: lrn_rate, model._images:data, model.labels: labels})

        if step < 10000:
            lrn_rate = 1e-4
        elif step < 20000:
            lrn_rate = 1e-6
        elif step < 30000:
            lrn_rate = 1e-8
        else:
            lrn_rate = 1e-10



        step += 1
        if step % 1 == 0:
            summary_writer.add_summary(summaries, train_step)
            print "Step: %d, Loss: %f"%(train_step, loss)
            summary_writer.flush()
            #print predictions.shape, labels.shape

        if step == 100000:
            break
    sv.Stop()

def evaluate(hps, start, end):
    """Eval loop."""
    #hps.batch_size = 1

    print "Eval"

    model = vgg_model.VGG(hps, FLAGS.mode)
    model.build_graph()
    saver = tf.train.Saver()
    #summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    try:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
        return

    print 'Loading checkpoint %s', ckpt_state.model_checkpoint_path
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    count = 1
    for folder in range(1,30):
        sub_start = (folder-1)*200+1
        for video in range(sub_start, sub_start+200):
            dir_path = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/%03d/K_%05d/"%(folder, video)
            depth_images = get_all_frames(dir_path)

            dest_dir = dir_path.replace("valid","heat_maps/valid")
            try:
                os.makedirs(dest_dir)
            except:
                pass

            predictions = []

            for i in range(0,depth_images.shape[0],50):
                predictions.append(sess.run(model.predictions, feed_dict={model._images: depth_images[i:i+50,:,:,:]}))
            predictions = np.vstack(predictions)
            print predictions.shape
            np.save(os.path.join(dest_dir,"heat_map.npy"), predictions)
        continue
        raw_input()
        for i in range(depth_images.shape[0]):
            image = depth_images[i:i+1,:,:,:]
            color_image = color_images[i,:,:,:]
            print i, image.shape, color_image.shape
            dest_path = os.path.join(dest_dir,"%d.jpg"%(i+1))
            print dest_path



            predictions = sess.run(model.predictions, feed_dict={model._images: image})[0]
            print np.min(predictions), np.max(predictions)

            image = image[0,:,:,0]

            predictions = resize(predictions, (240,320,2), order=3)

            plt.clf()

            plt.subplot(231)
            plt.imshow(color_image)

            plt.subplot(232)
            plt.imshow(image, cmap="gray")

            plt.subplot(233)
            plt.imshow(image, cmap="gray")

            plt.subplot(234)
            plt.imshow(image, cmap="gray")
            plt.imshow(predictions[:, :, 0], alpha=0.5)
            plt.colorbar()

            plt.subplot(235)
            plt.imshow(image, cmap="gray")
            plt.imshow(predictions[:, :, 1], alpha=0.5)
            plt.colorbar()

            colors = ['r','b']

            for part in range(2):

                heat_map = predictions[:, :, part]
                smoothed_map = gaussian_filter(heat_map, sigma=3)
                peaks = find_peaks(smoothed_map)

                print "Peaks ",peaks
                plt.subplot(233)
                if len(peaks)>0:
                    plt.plot(peaks[1],peaks[0],'o',color=colors[part])
                plt.xlim([0,320])
                plt.ylim([240,0])
            #plt.show()
            plt.savefig(dest_path)


def get_random_image():
    mean_image = np.load("/s/red/a/nobackup/cwc/palm_recognition/dataset/mean.npy")
    root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/%03d/"

    while True:
        video = random.randint(2401,2600)
        folder = ((video - 1) / 200) + 1
        frame = random.randint(1, len(os.listdir(os.path.join(root, "K_%05d") % (folder, video))))
        image_path = os.path.join(root, "K_%05d", "%d.jpg") % (folder, video, frame)
        image = skimage.io.imread(image_path,True)

        image -= mean_image

        image = image[np.newaxis,:,:,np.newaxis]

        color_image = skimage.io.imread(image_path.replace("K","M"))

        yield image, color_image

def get_all_frames(dir_path):
    mean_image = np.load("/s/red/a/nobackup/cwc/palm_recognition/dataset/mean.npy")

    num_frames = len(os.listdir(dir_path))
    depth_images = []
    #color_images = []
    for frame in range(1, num_frames+1):
        image_path = os.path.join(dir_path, "%d.jpg"%frame)
        image = skimage.io.imread(image_path,True)

        image -= mean_image

        image = image[np.newaxis,:,:,np.newaxis]

        #color_image = skimage.io.imread(image_path.replace("K","M"))

        depth_images.append(image)
    #color_images.append(color_image)


    depth_images = np.vstack(depth_images)
    #color_images = np.stack(color_images)
    print dir_path, depth_images.shape#, color_images.shape
    return depth_images#, color_images

def find_peaks(heat_map):
    map_left = np.zeros(heat_map.shape)
    map_left[1:, :] = heat_map[:-1, :]
    map_right = np.zeros(heat_map.shape)
    map_right[:-1, :] = heat_map[1:, :]
    map_up = np.zeros(heat_map.shape)
    map_up[:, 1:] = heat_map[:, :-1]
    map_down = np.zeros(heat_map.shape)
    map_down[:, :-1] = heat_map[:, 1:]

    peaks_binary = np.logical_and.reduce((heat_map >= map_left, heat_map >= map_right, heat_map >= map_up, heat_map >= map_down, heat_map > 0.1))
    return np.nonzero(peaks_binary)


def test(hps):
    """Eval loop."""


    image_iterator = get_random_image()

    model = vgg_model.VGG(hps, FLAGS.mode)
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    count = 1
    while True:
        #time.sleep(30)
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            print 'No model to eval yet at %s', FLAGS.log_root
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        print 'Loading checkpoint %s', ckpt_state.model_checkpoint_path
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        image, color_image = image_iterator.next()
        print image.shape

        predictions = sess.run(model.predictions, feed_dict={model._images: image})[0]

        image = image[0,:,:,0]

        predictions = resize(predictions, (240,320,2), order=3)



        plt.subplot(231)
        plt.imshow(color_image)

        plt.subplot(232)
        plt.imshow(image, cmap="gray")

        plt.subplot(233)
        plt.imshow(image, cmap="gray")

        plt.subplot(234)
        plt.imshow(image, cmap="gray")
        plt.imshow(predictions[:, :, 0], alpha=0.5)
        plt.colorbar()

        plt.subplot(235)
        plt.imshow(image, cmap="gray")
        plt.imshow(predictions[:, :, 1], alpha=0.5)
        plt.colorbar()

        colors = ['r','b']

        for part in range(2):

            heat_map = predictions[:, :, part]
            smoothed_map = gaussian_filter(heat_map, sigma=3)
            peaks = find_peaks(smoothed_map)

            print "Peaks ",peaks
            plt.subplot(233)
            if len(peaks)>0:
                plt.plot(peaks[1],peaks[0],'o',color=colors[part])
            plt.xlim([0,320])
            plt.ylim([240,0])

        plt.show()

def main(_):

    FLAGS.log_root = "/s/red/a/nobackup/cwc/tf/palm/"
    if FLAGS.mode == 'test' or FLAGS.mode=='eval':
        batch_size = 1
    else:
        batch_size = 32
    num_classes = 2

    image_shape = (240,320,1)


    hps = vgg_model.HParams(batch_size=batch_size,
                            image_shape=image_shape,
                            num_classes=num_classes,
                            min_lrn_rate=0.0001,
                            lrn_rate=0.1,
                            num_residual_units=5,
                            use_bottleneck=True,
                            weight_decay_rate=0.0002,
                            relu_leakiness=0,
                            optimizer='adam')

    if FLAGS.mode == 'train':
        FLAGS.train_dir = "%s/train"%FLAGS.log_root
        train(hps)
    elif FLAGS.mode == 'eval':
        #FLAGS.eval_dir = "%s/eval"%FLAGS.log_root
        start = 163
        end = 165
        evaluate(hps,start,end)

    elif FLAGS.mode == 'test':
        test(hps)


if __name__ == '__main__':
    #get_all_frames("/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/train/001/K_00001/")
    tf.app.run()
