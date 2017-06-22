# right-elbow: 3; right-wrist:4; left-elbow: 6; left-wrist:7
Right_Elbow = 3
Right_Wrist = 4
Left_Elbow = 6
Left_Wrist = 7

import numpy as np
import os
from sklearn.decomposition import PCA
#import cv2

#wk_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/'
#ske_dir = wk_dir + 'skeleton/'
result_dir = './result_valid/'

wind_sz = 10

#def readWrists():


def diffWrists(left_wrist, right_wrist): 

	diff_wrist = []
	for ind in range(len(left_wrist)):
		if len(left_wrist[ind]) > 0 and len(right_wrist[ind]) > 0:
			diff_wrist.append(np.array(left_wrist[ind]) - np.array(right_wrist[ind]))
		else:
			diff_wrist.append([])
			
	return np.array(diff_wrist)

def det_window(mid_ind, wrist, wind_sz, verbose = False):
	#frames = wrist[:, 3].tolist()
	right_frames = wind_sz // 2 # number of frames to the right of mid_ind
	left_frames = wind_sz - right_frames - 1 # number of frames to the left of mid_ind
	start_ind = (mid_ind - left_frames) if (mid_ind - left_frames) > 0 else 0
	end_ind = (mid_ind + right_frames) if (mid_ind + right_frames < len(wrist)) else len(wrist) - 1
	
	wrist_wind = []
	ind_wind = []
	for i in range(start_ind,end_ind+1):
		if len(wrist[i]) > 0:
			wrist_wind.append(wrist[i])
			ind_wind.append(i)
	if verbose:
		print(ind_wind)
		
	return ind_wind, np.array(wrist_wind)

def test_detWindow():
	testCases = [list(range(1, 30)), list(range(1,50,2)), list(range(5,50,2)), list(range(5,50,3)), list(range(5,50,10))]
	for frames in testCases:
		print(frames,'\n')
		for mid_ind in range(len(frames)):
			(start_ind, end_ind) = det_window(mid_ind, frames, wind_sz)
			print('\t%s:%s' %(frames[mid_ind], frames[start_ind:end_ind + 1]))
			print()

def neckMean(neck):
	neck_mean = []
	for i in range(len(neck)):
		_, neck_wind = det_window(i, neck, wind_sz)
		if len(neck_wind) > 0:
			neck_mean.append(np.mean(neck_wind, axis = 0))
		else:
			neck_mean.append([np.nan, np.nan, np.nan])

	return np.array(neck_mean)

def normalize(wrist, verbose = False):
	#left_wrist = np.array(left_wrist)
	#right_wrist = np.array(right_wrist)
	# Frame number remains unchanged
	#if len(wrist) >= 4: 
	wrist_mean = np.mean(wrist, axis = 0)
	wrist_std = np.std(wrist, axis = 0)
	if all(wrist_std != 0.0):
		norm_wrist = (wrist - wrist_mean)/wrist_std
	else:
		norm_wrist = []
	if verbose:
		print('\tmean: ', wrist_mean)
		print('\tstandard deviation: ', wrist_std)
		print('\tNormalized result: ', norm_wrist)
	#else:
	#	norm_wrist = []
	#	wrist_mean = []

	return wrist_mean, np.array(norm_wrist)

def pca_helper(norm_wind, verbose = False):
	
	cov = np.dot(norm_wind.T, norm_wind)
	_, eVal, eVec = np.linalg.svd(cov)
	
	if verbose:
		print('\t%s' %eVec)
		print('\t%s' %eVal)
	
	return eVec, eVal


def pathLength(eVec, norm_wind):
	pl = [0,0]
	for d in range(2):
	#pl = np.array([[0],[0],[0]])
		for t in range(len(norm_wind) - 1):
			pl[d] += np.absolute(np.inner(eVec[d], (norm_wind[t+1] - norm_wind[t])))
	return np.array(pl)
	
def signedPathNoAbs(eVec, norm_wind):
	sp = [0,0]
	for d in range(2):
		for t in range(len(norm_wind) - 1):
			sp[d] = sp[d] + np.inner(eVec[d], (norm_wind[t+1] - norm_wind[t]))
	return np.array(sp)
		
if __name__ == '__main__':
	# test det_window() function
	# test_detWindow()

	folders = sorted(os.listdir(result_dir))
	for ea_folder in folders:
		if ea_folder <= '024':
			continue
		# if ea_folder > '024':
		# 	break
		folder_dir_reading = result_dir + ea_folder + '/' 
		print('Reading folder %s' %ea_folder)

		videos = sorted(os.listdir(folder_dir_reading))	
		for ea_video in videos:
			# if ea_video == 'M_00005':
			# 	break
			vid_dir_reading = folder_dir_reading + ea_video + '/'
			print('\tReading video %s' %ea_video)
			
			left_wrist = np.array(np.load(vid_dir_reading + 'left_wrist.npy'))
			right_wrist = np.array(np.load(vid_dir_reading + 'right_wrist.npy'))
			neck = np.array(np.load(vid_dir_reading + 'neck.npy'))
			diff_wrist = diffWrists(left_wrist, right_wrist)#, framesLeft, framesRight)
			#print(diff_wrist)
			
			neck_mean = neckMean(neck) # the mean position of neck
			#print(neck_mean)
			# norm_left = normalize(left_wrist)
			# norm_right = normalize(right_wrist)
			# norm_diff = normalize(diff_wrist)
			
			wrists = [left_wrist, right_wrist, diff_wrist]
			#frames = [framesLeft, framesRight, framesDiff]
			names = ['LEFT', 'RIGHT', 'DIFF']
			
			final = []			
			for mid_ind in range(len(left_wrist)):
				#print('frame %d' %(mid_ind + 1))

				for h in range(3):
					#print(names[h])
					ind_wind, wrist_wind = det_window(mid_ind, wrists[h], wind_sz)
					if len(wrist_wind) > 0:
						wrist_mean, norm_wind = normalize(wrist_wind)
					else:
						wrist_mean = np.array([np.nan, np.nan, np.nan])
						norm_wind = []
					
					# Cacluate features
					
					#Feature #1-3
					norm_mean = wrist_mean - neck_mean[mid_ind] if h != 2 else wrist_mean # [x,y,d] mean for each window per wrist, normalized to neck position; Feature #1-3
					#print(norm_mean)
					
					#Feature #4-16
					if len(norm_wind) >= 3: # less than 3 samples will give two 0 igenvalues 
						eVec, eVal = pca_helper(norm_wind)
						#print(pca.explained_variance_)
						#if len(pca.explained_variance_) == 3: 
						e1 = eVec[0] # First eigenvector; Feature #4-6
						e2 = eVec[1] # Second eigenvector; Feature #7-9
						lambs = eVal[:] #Three eigenvalues; Feature #10-12

						pl = pathLength(eVec, norm_wind)
						#print(pl)
						sp = signedPathNoAbs(eVec, norm_wind)[0:2]
						sp = np.absolute(sp) if h != 2 else sp
						# print(h)
						# print(sp)
						
						linearity = 1.5 * (lambs[0]/np.sum(lambs) - 1.0/3) # linearity: Feature #13
						planarity = 3*((lambs[0] + lambs[1])/np.sum(lambs)) - 2.0 # planarity: Feature #14
						monotonicity = sp/pl # monotonicity: Feature #15-16
							
					else:					
						e1 = [np.nan, np.nan, np.nan]
						e2 = [np.nan, np.nan, np.nan]
						lambs = [np.nan, np.nan, np.nan]
						linearity = np.nan
						planarity = np.nan
						monotonicity = [np.nan, np.nan]
					
					combined = [norm_mean, e1, e2, lambs, linearity, planarity, monotonicity]
					
					if h == 0:
						final.append([combined, [], []])
					elif h == 1: 
						final[mid_ind][1] = combined
						e1L = final[mid_ind][0][1] # left wrist eigenvector 1
						e2L = final[mid_ind][0][2] # left wrist eigenvector 2
						# codirecitonality: feature 49-50
						final[mid_ind].append([np.inner(e1, e1L), np.inner(e2, e2L)])
					else:
						final[mid_ind][2] = combined
			# print(final)
			np.save(vid_dir_reading + 'features.npy', final)
					
			# Finally codirectionality: Feature 49-50
			#for key in final:
			#	if len(final[key][0]) > 0 and len(final[key]) 


	# result = pca_helper(norm_wrist)
	# 			#print(result)
	# 			# Result structure: [[[[EigenVector1], [EigenVector2]], [EigenValue]], [Window2], ...]
	# 			# result is [] if there is NaN entry in norm_wrist
	# 			np.save(vid_dir_reading + wrist + '_pca.npy', result) 
		


