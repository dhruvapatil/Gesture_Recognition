import rgb_resnet_model
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class RealTimeHandRecognition():
    def __init__(self, hands, gestures):

        hps = rgb_resnet_model.HParams(batch_size=None,
                                                num_classes=gestures,
                                                min_lrn_rate=0.0001,
                                                lrn_rate=0.1,
                                                num_residual_units=5,
                                                use_bottleneck=True,
                                                weight_decay_rate=0.0002,
                                                relu_leakiness=0,
                                                optimizer='mom')


        width = 100
        height = 100

        model = rgb_resnet_model.ResNet(hps, "eval", width, height)
        model.build_graph()
        saver = tf.train.Saver()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tf.train.start_queue_runners(sess)
        ckpt_state = tf.train.get_checkpoint_state("/s/red/a/nobackup/cwc/tf/chalearn/rgb/%s"%hands)
        print 'Loading checkpoint %s', ckpt_state.model_checkpoint_path
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        self.sess = sess
        self.model = model

        self.past_probs = None

    def classify(self, data):
        predictions = []
        for i in range(0, data.shape[0], 100):
            predictions.append(
                self.sess.run(self.model.predictions, feed_dict={self.model._images: data[i:i + 100, :, :, :]}))
        predictions = np.vstack(predictions)
        return predictions


    def features(self, data):
        features = []
	for i in range(0,data.shape[0],100):
	    features.append(self.sess.run(self.model.features, feed_dict={self.model._images: data[i:i+100,:,:,:]}))
	features = np.vstack(features)
        return features
