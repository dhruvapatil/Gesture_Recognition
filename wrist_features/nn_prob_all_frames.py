import numpy as np
import tensorflow as tf
import os
from collections import Iterable
import features as fe

Right_Elbow = 3
Right_Wrist = 4
Left_Elbow = 6
Left_Wrist = 7
Neck = 1

nn_run_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/NN_jason/0621_withDepth/'
result_dir = './result_valid/'
#result_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/wrist_features/withDepth/'
#prob_dir = './nn_prob_tmp/'
prob_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/NN_jason/nn_prob_validation_withDepth/'

#-------NN start------
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 50])
W_fc1 = tf.Variable(tf.truncated_normal([50, 100], stddev = 0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [100]))

h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = tf.Variable(tf.truncated_normal([100, 200], stddev = 0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape = [200]))

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = tf.Variable(tf.truncated_normal([200, 249], stddev = 0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape = [249]))

y = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

saver = tf.train.Saver()

saver.restore(sess, nn_run_dir + 'model.ckpt')
print('model restored')
#-------NN end-------

folders = sorted(os.listdir(result_dir))
for ea_folder in folders:
	# if ea_folder <= '024':
	# 	continue
	# if ea_folder > '024':
	# 	break
	folder_dir_reading = result_dir + ea_folder + '/' 
	print('Reading folder %s' %ea_folder)
	
	videos = sorted(os.listdir(folder_dir_reading))	
	for ea_video in videos:
		vid_dir_reading = folder_dir_reading + ea_video + '/'
		
		frames = np.load(vid_dir_reading + 'features.npy', encoding = 'latin1')
		x_test = []
		for frame in frames:
			x_test.append(list(flatten(frame.tolist())))

		probs = sess.run(y, {x: x_test})
		svDir = prob_dir + ea_folder + '/'
		if not os.path.exists(svDir):
			os.makedirs(svDir)
		np.save(svDir + ea_video + '_withDepth.npy', probs)
		print('\t%s Done' %ea_video)
		
		
		
			
			
			
