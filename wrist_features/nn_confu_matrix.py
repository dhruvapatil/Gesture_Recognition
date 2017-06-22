import numpy as np
import tensorflow as tf

#wk_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/'
#ske_dir = wk_dir + 'skeleton/'
# 249 classes
result_dir = './NN_jason/'
list_file = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/train_list.txt'
nn_run_dir = './NN_jason/0621_withDepth/'

def readData(verbose = True):
	x_test = np.load('./SVM/x_test.npy')
	y_test = np.load('./SVM/y_test.npy') - 1
	#y_test = tf.one_hot(y_test, 249)

	if verbose:
		print(x_test.shape)
		print(y_test.shape)
		
	return x_test, y_test

def batch_iter(data, labels, batch_size, num_epochs, shuffle=False):

	data_size = data.shape[0]
	num_batches_per_epoch = int((data_size-1)/batch_size) + 1
	for epoch in range(num_epochs):
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield data[start_index:end_index], labels[start_index:end_index]

if __name__ == '__main__':
	#read data
	x_test, y_test = readData(True)

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
	y = tf.argmax(y, axis = 1)
	y_ = tf.placeholder(tf.int64, [None])

	#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(y, y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#tf.summary.scalar('accuracy', accuracy)
	#tf.summary.scalar('loss', cross_entropy)
	#merged = tf.summary.merge_all()
	#train_writer = tf.summary.FileWriter(nn_run_dir + '/train', sess.graph)
	#test_writer = tf.summary.FileWriter(nn_run_dir + '/test', sess.graph)
	#saver = tf.train.Saver(max_to_keep = 10)

	#sess.run(tf.global_variables_initializer())

	#y_train = sess.run(y_train)
	#y_test = sess.run(y_test)
	saver = tf.train.Saver()
	
	saver.restore(sess, nn_run_dir + 'model.ckpt')
	print('model restored')
	
	cMatrix = tf.confusion_matrix(y, y_)
	
	acc, cMatrix_res = sess.run([accuracy, cMatrix], feed_dict={x: x_test, y_: y_test})
	print(acc)
	print(cMatrix_res.shape)
	np.save(result_dir + 'confusion_matrix_0621_per_frame.npy', cMatrix_res)
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
