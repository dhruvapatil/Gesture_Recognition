# right-elbow: 3; right-wrist:4; left-elbow: 6; left-wrist:7
Right_Elbow = 3
Right_Wrist = 4
Left_Elbow = 6
Left_Wrist = 7
Neck = 1

import numpy as np
import os

wk_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/'
ske_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/skeleton/skeleton_valid/'
result_dir = './result_valid/'

folders = sorted(os.listdir(ske_dir))
for ea_folder in folders:
	#if ea_folder == '002':
	#	break
	folder_dir_reading = ske_dir + ea_folder + '/'
	folder_dir_writing = result_dir + ea_folder + '/'
	if not os.path.exists(folder_dir_writing):
		os.makedirs(folder_dir_writing)
 
	print('Reading folder %s' %ea_folder)
	videos = sorted(os.listdir(folder_dir_reading))	
	for ea_video in videos:
		#if ea_video == 'M_00003':
		#	break
		vid_dir_reading = folder_dir_reading + ea_video + '/'
		vid_dir_writing = folder_dir_writing + ea_video + '/'
		if not os.path.exists(vid_dir_writing):
			os.makedirs(vid_dir_writing)

		print('\tReading video folder %s' %ea_video)
		frame_no = len(os.listdir(vid_dir_reading)) // 3
		left_wrist = []
		right_wrist = []
		neck = []
		for frame_ind in range(frame_no):
			peaks = np.load(vid_dir_reading + str(frame_ind+1) + '_peaks.npy')
			#if frame_ind == 1:
			#	print(peaks)
			#for i in range(len(peaks)):
			if len(peaks[Right_Wrist]) > 0:
				right_wrist.append(list(peaks[Right_Wrist][0][0:3]))
			else:
				right_wrist.append([])
			if len(peaks[Left_Wrist]) > 0:
				left_wrist.append(list(peaks[Left_Wrist][0][0:3]))
			else:
				left_wrist.append([])
			if len(peaks[Neck]) > 0:
				neck.append(list(peaks[Neck][0][0:3]))
			else:
				neck.append([])
		#print(len(left_wrist))
		#for x in left_wrist:
			#print(x)
		#print(len(right_wrist))
		#for x in right_wrist:
			#print(x)
		#print(len(neck))
		#for x in neck:
			#print(x)	
		# #print(left_wrist)
		# for y in right_wrist:
		# 	print(y)
		#print(right_wrist)
		#print('\tSaving video folder %s' %ea_video)
		#print(neck)
		#print(left_wrist)
		np.save(vid_dir_writing + 'left_wrist.npy', left_wrist)
		np.save(vid_dir_writing + 'right_wrist.npy', right_wrist)
		np.save(vid_dir_writing + 'neck.npy', neck)
		#print('\t')



