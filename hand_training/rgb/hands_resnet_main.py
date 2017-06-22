import numpy as np
import os
import tensorflow as tf

import hands_resnet_model
from image_reader import get_input

os.environ["CUDA_VISIBLE_DEVICES"]="0"
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('hand', 'right', 'right or left')
tf.app.flags.DEFINE_string('mode', 'train', 'train')
tf.app.flags.DEFINE_integer('image_size', 100, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')



def train(hps):
    """Training loop."""

    images, labels = get_input(hps.batch_size, FLAGS.hand)

    model = hands_resnet_model.ResNet(hps, FLAGS.mode, images, labels)
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

    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)

    for var, val in zip(tvars, tvars_vals):
        print var.name, val, val.shape

    #raw_input()

    step = 0
    lrn_rate = 0.1


    while not sv.should_stop():

        print step,
        (_, summaries, loss, predictions, truth, train_step) = sess.run(
            [model.train_op, model.summaries, model.cost, model.predictions,
             model.labels, model.global_step],
            feed_dict={model.lrn_rate: lrn_rate})

        if train_step < 10000:
            lrn_rate = 0.1
        elif train_step < 20000:
            lrn_rate = 0.01
        elif train_step < 30000:
            lrn_rate = 0.001
        else:
            lrn_rate = 0.0001

        truth = np.argmax(truth, axis=1)
        predictions = np.argmax(predictions, axis=1)
        precision = np.mean(truth == predictions)

        step += 1
        if step % 100 == 0:
            print train_step, precision
            #print "\n",np.min(sess.run(model.fc_x)),np.min(sess.run(model.fc_b)),np.min(sess.run(model.fc_w))
            precision_summ = tf.Summary()
            precision_summ.value.add(
                tag='Precision', simple_value=precision)
            summary_writer.add_summary(precision_summ, train_step)
            summary_writer.add_summary(summaries, train_step)
            tf.logging.info('loss: %.3f, precision: %.3f\n' % (loss, precision))
            summary_writer.flush()
        #if step == 100000:
        #    break'''
    sv.Stop()


def main(_):

    if FLAGS.hand == "right":
        FLAGS.log_root = "/s/red/a/nobackup/cwc/tf/chalearn/rgb/RH"
        FLAGS.train_dir = "/s/red/a/nobackup/cwc/tf/chalearn/rgb/RH/train"

    elif FLAGS.hand == "left":
        FLAGS.log_root = "/s/red/a/nobackup/cwc/tf/chalearn/rgb/LH"
        FLAGS.train_dir = "/s/red/a/nobackup/cwc/tf/chalearn/rgb/LH/train"
    else:
        return

    print FLAGS.log_root

    batch_size = 128
    num_classes = 250

    hps = hands_resnet_model.HParams(batch_size=batch_size,
                                     num_classes=num_classes,
                                     min_lrn_rate=0.0001,
                                     lrn_rate=0.1,
                                     num_residual_units=5,
                                     use_bottleneck=True,
                                     weight_decay_rate=0.0002,
                                     relu_leakiness=0,
                                     optimizer='mom')

    train(hps)


if __name__ == '__main__':
    tf.app.run()
