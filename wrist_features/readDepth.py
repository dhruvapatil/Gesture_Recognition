from skimage.transform import rescale
import skimage.io
import numpy as np
import os
import matplotlib.pyplot as plt
import random

Right_Elbow = 3
Right_Wrist = 4
Left_Elbow = 6
Left_Wrist = 7
Neck = 1


train_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/'
result_dir = './result_valid/'

def register(imgK, imgM, frame_no, verbose = False):
	depth_image_path = os.path.join(imgK, "%d.jpg") % (frame_no+1)
	rgb_image_path = os.path.join(imgM, "%d.jpg") % (frame_no+1)
	
	fixedImage = rescale(skimage.io.imread(depth_image_path, True), 0.92)
	movingImage = skimage.io.imread(rgb_image_path, True)

	f1 = np.zeros(movingImage.shape)
	f1[19:, 20:20 + 294] = fixedImage

	if verbose:
		print(f1.shape)
		print(movingImage.shape)
		plt.subplot(2,2,1)
		plt.imshow(movingImage, cmap='gray')
		plt.imshow(fixedImage, cmap='gray', alpha=0.7)
		
		plt.subplot(2,2,2)
		plt.imshow(movingImage, cmap='gray')
		plt.imshow(f1, cmap='gray', alpha=0.7)

		plt.subplot(2,2,3)
		plt.imshow(movingImage, cmap='gray')

		plt.subplot(2,2,4)
		plt.imshow(f1, cmap='gray')
		plt.show()

	return f1

def readZ(reg, frameInfo):
	xlim, ylim = reg.shape

	if len(frameInfo) > 0 and int(frameInfo[0]) < xlim and int(frameInfo[1]) < ylim:
		frameInfo[2] = reg[int(frameInfo[0])][int(frameInfo[1])]
		return frameInfo
	else:
		return []

def testReadZ(reg, imgM, frameInfo, frame_no):
	if len(frameInfo) == 0:
		return
	plt.subplot(1,2,1)
	plt.imshow(reg, cmap='gray')
	plt.plot(frameInfo[0], frameInfo[1],'g.', markersize=10.0)

	rgb_image_path = os.path.join(imgM, "%d.jpg") % (frame_no+1)
	movingImage = skimage.io.imread(rgb_image_path, True)

	plt.subplot(1,2,2)
	plt.imshow(movingImage, cmap='gray')
	plt.plot(frameInfo[0], frameInfo[1],'g.', markersize=10.0)
	plt.show()

def testReadData():
	folders = sorted(os.listdir(result_dir))
	ea_folder = random.choice(folders)
	folder_dir_img = train_dir + ea_folder + '/'
	folder_dir_res = result_dir + ea_folder + '/'
	print('Reading folder %s' %ea_folder)

	videos = sorted(os.listdir(folder_dir_res))	
	ea_video = random.choice(videos)
	vid_dir_imgM = folder_dir_img + ea_video + '/'
	vid_dir_res = folder_dir_res + ea_video + '/'
	vid_dir_imgK = folder_dir_img + ea_video.replace('M','K') + '/'
	print('\tReading video folder %s' %ea_video)

	left_wrist = np.load(vid_dir_res + 'left_wrist.npy')
	right_wrist = np.load(vid_dir_res + 'right_wrist.npy')
	neck = np.load(vid_dir_res + 'neck.npy')
	
	samples = [left_wrist, right_wrist, neck]
	names = ['left','right','neck']

	frame_no = random.randrange(len(left_wrist))
	reg = register(vid_dir_imgK, vid_dir_imgM, frame_no, False) #registed depth image
	for i in range(3):
		
		frame = samples[i][frame_no]
		print('-------------------')
		print(ea_video + ': ' + str(frame_no) + ': ' + names[i])
		print(reg.shape)
		print(frame)
		frame = readZ(reg, frame)
		print(samples[i][frame_no])
		testReadZ(reg, vid_dir_imgM, frame, frame_no)


def readData(test = False, test2 = False, half = True):
	folders = sorted(os.listdir(result_dir))
	for ea_folder in folders:
		# if half and ea_folder <= '015':
		# 	continue
		if half and ea_folder <= '024':
			continue
		if half and ea_folder == '026':
			break
		if test and ea_folder == '002':
			break
		folder_dir_img = train_dir + ea_folder + '/'
		folder_dir_res = result_dir + ea_folder + '/'
		print('Reading folder %s' %ea_folder)

		videos = sorted(os.listdir(folder_dir_res))	
		for ea_video in videos:
			if test and ea_video == 'M_00002':
				break
			vid_dir_imgM = folder_dir_img + ea_video + '/'
			vid_dir_res = folder_dir_res + ea_video + '/'
			vid_dir_imgK = folder_dir_img + ea_video.replace('M','K') + '/'
			print('\tReading video folder %s' %ea_video)

			
			left_wrist = np.load(vid_dir_res + 'left_wrist.npy')
			right_wrist = np.load(vid_dir_res + 'right_wrist.npy')
			neck = np.load(vid_dir_res + 'neck.npy')
			
			samples = [left_wrist, right_wrist, neck]
			names = ['left','right','neck']
			if test2:
				print(left_wrist)
			for frame_no in range(len(left_wrist)):
				reg = register(vid_dir_imgK, vid_dir_imgM, frame_no, False) #registed depth image
				for i in range(3):
					frame = samples[i][frame_no]
					if test:
						print('-------------------')
						print(ea_video + ': ' + str(frame_no) + ': ' + names[i])
						print(reg.shape)
						print(samples[i][frame_no])
					frame = readZ(reg, frame)
					if test:
						print(samples[i][frame_no])
						testReadZ(reg, vid_dir_imgM, frame, frame_no)
			if test2:
				print(left_wrist)
			
			np.save(vid_dir_res + 'left_wrist.npy', left_wrist)
			np.save(vid_dir_res + 'right_wrist.npy', right_wrist)
			np.save(vid_dir_res + 'neck.npy', neck)
			

if __name__ == '__main__':
	readData()


