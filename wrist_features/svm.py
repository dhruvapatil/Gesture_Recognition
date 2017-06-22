# right-elbow: 3; right-wrist:4; left-elbow: 6; left-wrist:7
Right_Elbow = 3
Right_Wrist = 4
Left_Elbow = 6
Left_Wrist = 7

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import svm
import cv2
from collections import Iterable
import pickle

#wk_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/'
#ske_dir = wk_dir + 'skeleton/'
result_dir = './result/'
list_file = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/train_list.txt'

def flatten(lis):
	for item in lis:
		if isinstance(item, Iterable) and not isinstance(item, basestring):
			for x in flatten(item):
				yield x
		else:
			if np.isnan(item):
				yield 0
			else:
				yield item

def readData(verbose = False):
	f = open(list_file, 'r')
	
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	folders = sorted(os.listdir(result_dir))
	for ea_folder in folders:
		#if ea_folder > '100':
		#	break
		folder_dir_reading = result_dir + ea_folder + '/'
		
		if ea_folder <= '160':
			x = x_train
			y = y_train
		else:
			x = x_test
			y = y_test
		videos = sorted(os.listdir(folder_dir_reading))	
		for ea_video in videos:
			#if ea_video == 'M_00002':
			#	break
			print('Reading video: %s' %ea_video)
			s = f.readline()
			label = int(s.split()[-1])
			
			vid_dir_reading = folder_dir_reading + ea_video + '/'			
			features = np.load(vid_dir_reading + 'features.npy')
			for frame in features:
				x.append([])
				x[-1] = list(flatten(frame.tolist()))
				y.append(label)
	f.close()
	if verbose:
		#print(x_train)
		print(len(x_train))
		#print(y_train)
		print(len(x_test))
		
	return x_train, y_train, x_test, y_test
					

if __name__ == '__main__':
	x_train, y_train, x_test, y_test = readData(True)
	np.save('./SVM/x_train.npy', x_train)
	np.save('./SVM/y_train.npy', y_train)
	np.save('./SVM/x_test.npy', x_test)
	np.save('./SVM/y_test.npy', y_test)
	print('Data Saved')
	
	#x_train = [[0,1],[2,5]]
	#y_train = [1,2]
	#x_test = [[0,1],[2,5]]
	#y_test = [1,3]
	
	# cl = svm.SVC()
	
	# f = open('./SVM/model.npy', 'w')
	# pickle.dump(cl, f)
	# f.close()
	# print('Model Saved')
	
	# #f = open('./SVM/model.npy', 'r')
	# #cl = pickle.load(f)
	
	# cl.fit(x_train, y_train)
	# pred = cl.predict(x_test)
	# acc = sum(pred==y_test) * 1.0/len(y_test)
	# #f.close()
	
	# f = open('./SVM/acc.txt', 'w')
	# f.write(str(acc))
	# f.close()
	
	
	
	
	
