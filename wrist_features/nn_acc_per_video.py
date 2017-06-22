import numpy as np
import tensorflow as tf
import os
from collections import Iterable

#wk_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/'
#ske_dir = wk_dir + 'skeleton/'
# 249 classes
#result_dir = './NN_jason/'
list_file = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/train_list.txt'
ske_file = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/skeleton/skeleton_valid/'
nn_run_dir = './NN_jason/0621_withDepth/'
result_dir = './result/'
nn_dir = './NN_jason/'

def flatten(lis):
	for item in lis:
		if isinstance(item, Iterable) and not isinstance(item, str):
			for x in flatten(item):
				yield x
		else:
			if np.isnan(item):
				yield 0
			else:
				yield item


if __name__ == '__main__':
	#read network
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

	y_frame = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
	y_video = tf.argmax(tf.reduce_sum(y_frame, 0), axis = 0)

	y_ = tf.placeholder(tf.int64, [])

	saver = tf.train.Saver()
	
	saver.restore(sess, nn_run_dir + 'model.ckpt')
	print('model restored')
	
	#cMatrix = tf.confusion_matrix(y, y_)
	correct = tf.equal(y_video, y_)
	
	
	numVideo = 0
	numCorrect = 0
	f = open(list_file, 'r')
	res = open(nn_dir + 'acc_per_video_0621.txt', 'w')
	Truth = []
	Predicts = []
	#read test data
	folders = sorted(os.listdir(result_dir))
	for ea_folder in folders:
		#if ea_folder > '100':
		#	break
		folder_dir_reading = result_dir + ea_folder + '/'
		print('Reading folder: %s' %ea_folder)
		videos = sorted(os.listdir(folder_dir_reading))	
		for ea_video in videos:

			s = f.readline()
			if ea_folder <= '160':
				continue
				
			y_test = int(s.split()[-1]) - 1
			Truth.append(y_test)
			vid_dir_reading = folder_dir_reading + ea_video + '/'			
			features = np.load(vid_dir_reading + 'features.npy', encoding='latin1')
			x_test = []
			for frame in features:
				x_test.append(list(flatten(frame.tolist())))

			video_correct, predict = sess.run([correct, y_video], feed_dict={x: x_test, y_: y_test})
			Predicts.append(predict)
			numVideo += 1
			if(video_correct):
				print('\t' + ea_video)
				numCorrect += 1
				print('\tAccuracy: %s' %(numCorrect/numVideo))
			res.write(ea_video + '\tTruth:%s\tPredict:%s\n' %(y_test, predict))
	
	y_p = tf.placeholder(tf.float32, [None])
	y_t = tf.placeholder(tf.float32, [None])
	
	cMatrix = tf.confusion_matrix(y_p, y_t)
	
	cMatrix_res = sess.run(cMatrix, feed_dict={y_p: Truth, y_t: Predicts})
	print(cMatrix_res.shape)
	np.save(nn_dir + 'confusion_matrix_0621_perVideo.npy', cMatrix_res)
	#for x in cMatrix_res:
	#	print(x)

	res.close()
	
	#print(cMatrix_res.shape)
	#np.save(result_dir + 'confusion_matrix_0618.npy', cMatrix_res)
	#print(cMatrix_res)

	#test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
	#print("test accuracy %g"%train_accuracy)
	#f_test.write("step %d, test accuracy %g\n"%(step-1, test_accuracy))
	#f_test.close()
	#f_train.close()
#       if current_step % FLAGS.checkpoint_every == 0:
#           path = saver.save(sess, checkpoint_prefix, global_step=current_step)
		# print("Saved model checkpoint to {}\n".format(path))
		



	#y_ = tf.placeholder()
