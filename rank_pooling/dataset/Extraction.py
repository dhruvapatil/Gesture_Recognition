import os
from itertools import chain
from os.path import join
import numpy as np
from core.Normalization import normalize_by_joint
from core.RankPooling import rank_pooling
from dependencies.Processing import (sorted_frames_by_substring, get_magnitude, divide_list)
from distribute.Misc import get_machine_count
from features.Velocity import calculate_velocity
from features.Distances import (calculate_distance_between_wrists, calculate_distance_from_spine, calculate_product_of_joints)

machine_count = get_machine_count()



def symmetricity_extraction(data_left, labels_left, data_right, labels_right):
    mag_left = get_magnitude(data_left)[0]
    mag_right = get_magnitude(data_right)[0]

    swap_flag = False

    if (mag_left > mag_right):
        # print 'magnitude of left greater than right:==>swapping left for right: '
        swap_flag = True
        video_data = np.copy(data_left)
        video_data[:, ::2] = -1 * video_data[:, ::2]
        labels = labels_left
    elif (mag_right > mag_left):
        # print 'magnitude of right greater than left:==> keeping right as is: '
        video_data = np.copy(data_right)
        labels = labels_right
    elif (mag_left == mag_right):
        if mag_right > 0:
            # print 'keeping right as is: '
            video_data = np.copy(data_right)
            labels = labels_right
        elif mag_right == 0:
            pass#continue

    return video_data, labels, swap_flag


def get_arm_joint_data(frames_list, dir_name, arm_index=0, normalized=True):
    if arm_index == 0:
        joint_index_list = [1, 3, 4]
        name = 'right'
    elif arm_index == 1:
        joint_index_list = [1, 6, 7]
        name = 'left'

    video_frame_joints = []
    frame_file_log = []
    labels = []

    for frame_file in frames_list:
        a = np.load(join(dir_name, frame_file))

        try:
            video_frame_joints.append(list(chain(*[[a[i][0][0], a[i][0][1]] for i in joint_index_list])))
            frame_file_log.append(frame_file)
            labels.append(frame_file.split('_')[0])
        except:
            continue  # skip the frame if incomplete data found

    try:
        video_frame_joints = np.vstack(video_frame_joints)
        labels = map(int, labels)

        if normalized:
            # normalizing by mid spine joint
            normalized_video = normalize_by_joint(video_frame_joints, 0, verbose=True)
            # removing shoulder mid joint from the data, dimensionality of data = 4
            normalized_video = normalized_video[:, 2:]
            # print labels
            return normalized_video, labels
        else:
            return video_frame_joints, labels
    except:
        #print 'cannot vstack data', dir_name
        return [], []


def get_arm_joint_data_inter(video_dir, arm_index = 0, normalized=True):
    #print join(video_dir, 'vid.npy')
    try:
        data = np.load(join(video_dir, 'vid.npy'))

        wrist_distance = calculate_distance_between_wrists(data)

        if arm_index == 0:
            name='right' #joints are [1,3,4]
            video_data = data[:,:6]
        elif arm_index == 1:
            name = 'left' #joints are [1,6,7]
            video_data = data[:, [0,1,6,7,8,9]]

        if ((video_data>-1).all(axis=1)).any() == False:  #All the frames for a particular joint are not found: skip the video
            #print 'cannot vstack data', video_dir
            return [], [], [], []

        #CASE NOT HANDLED: joint missing only in a particular frame
        #If any joint in any particular frame is missing, discard the frame
        else:
            labels = list(np.arange(video_data.shape[0]))
            #print 'labels is: ',len(labels)
            if normalized:
                normalized_video = normalize_by_joint(video_data, 0)#, verbose=True)

                distance_from_spine = calculate_distance_from_spine(normalized_video)

                normalized_video = normalized_video[:, 2:]
                return normalized_video, labels, distance_from_spine, wrist_distance
            else:
                distance_from_spine = calculate_distance_from_spine(video_data)
                return video_data, labels, distance_from_spine, wrist_distance
    except:
        return [], [], [], []


def extract_rank_pooling_features(main_folder_path, des_folder_path, list_index, category = 'train', approach = 'normal'):
    folder_list = os.listdir(main_folder_path)
    if category == 'train':
        folder_list = folder_list[:150]
    elif category== 'val':
        folder_list = folder_list[150:]
    elif category =='valid':
        folder_list = folder_list
        list_index = -1

    np.save(('../index_log_files/'+str(list_index)), folder_list)
    if list_index == -1:
        folder_list= folder_list
    else:
        folder_list = divide_list(folder_list, list_index, machine_count= machine_count)

    #print len(folder_list)


    total_missing_entries_left = 0
    total_missing_entries_right = 0
    swap_count = 0


    for folder in folder_list:
        #print 'folder is: ', folder
        if not os.path.exists(join(des_folder_path, folder)): os.mkdir(join(des_folder_path, folder))

        video_list = os.listdir(join(main_folder_path, folder))
        # print folder, video_list
        for video in video_list:
            #print 'video is: ', video
            dir_name = join(main_folder_path, folder, video)
            des_dir_name = join(des_folder_path, folder, video)
            if not os.path.exists(des_dir_name): os.mkdir(des_dir_name)

            frames_sorted = sorted_frames_by_substring(dir_name, substring_name='_peaks.npy')

            data_right, labels_right = get_arm_joint_data(frames_sorted, dir_name, arm_index=0)
            data_left, labels_left = get_arm_joint_data(frames_sorted, dir_name, arm_index=1)

            if approach=='normal':
                if (len(data_right) > 0):
                    rp_video_right = rank_pooling(data_right, labels_right, linear=False)
                    # np.save((des_dir_name+'/right.npy'), rp_video_right)
                else:
                    total_missing_entries_right += 1
                if (len(data_left) > 0):
                    rp_video_left = rank_pooling(data_left, labels_left, linear=False)
                    # np.save((des_dir_name+'/left.npy'), rp_video_left)
                else:
                    total_missing_entries_left += 1


            elif approach=='symmetric':
                video_data, labels, swap_flag = symmetricity_extraction(data_left, labels_left, data_right, labels_right)
                if swap_flag:
                    swap_count+= 1
                rp_video = rank_pooling(video_data, labels, linear=False)
                #np.save((des_dir_name+'/one_arm.npy'), rp_video)

    '''
    if approach == 'normal':
        print 'total_missing_entries_right: ', total_missing_entries_right
        print 'total_missing_entries_left: ', total_missing_entries_left
    elif approach == 'symmetric':
        print 'total_swap count: ', swap_count
    '''


def extract_rank_pooling_features_inter(main_folder_path, des_folder_path, list_index, category='train', velocity=True, approach='normal'):
    folder_list = os.listdir(main_folder_path)
    if category == 'train':
        folder_list = folder_list[:150]
    elif category == 'val':
        folder_list = folder_list[150:]
    elif category == 'valid':
        folder_list = folder_list
        #list_index = -1

    #if list_index == -1:
    #    folder_list = folder_list
    #else:
    folder_list = divide_list(folder_list, list_index, machine_count=machine_count)

    np.save(('../index_log_files/' + str(list_index)), folder_list)

    total_missing_entries_left = 0
    total_missing_entries_right = 0
    swap_count = 0

    for folder in folder_list:
        #print 'folder is: ', folder
        if not os.path.exists(join(des_folder_path, folder)): os.mkdir(join(des_folder_path, folder))

        video_list = os.listdir(join(main_folder_path, folder))
        # print folder, video_list
        for video in video_list:
            #print 'video is: ', video
            dir_name = join(main_folder_path, folder, video)
            des_dir_name = join(des_folder_path, folder, video)
            if not os.path.exists(des_dir_name): os.mkdir(des_dir_name)


            data_right, labels_right, distance_from_spine_right, wrist_distance_right = get_arm_joint_data_inter(dir_name, arm_index=0)
            data_left, labels_left, distance_from_spine_left, wrist_distance_left = get_arm_joint_data_inter(dir_name, arm_index=1)

            right_flag = left_flag = False

            if velocity:
                remove_frames = 1
                if len(data_right) > 2:
                    velocity_right = calculate_velocity(data_right)
                    data_right = data_right[remove_frames:-remove_frames, :]
                    data_right = np.hstack((data_right, velocity_right))

                    product_right = calculate_product_of_joints(data_right)

                    data_right = np.hstack((data_right, product_right, distance_from_spine_right, wrist_distance_right))

                    labels_right = labels_right[1:-1]
                    right_flag = True


                if len(data_left) > 2:
                    velocity_left = calculate_velocity(data_left)
                    data_left = data_left[remove_frames:-remove_frames, :]
                    data_left = np.hstack((data_left, velocity_left))

                    product_left = calculate_product_of_joints(data_left)

                    data_left = np.hstack((data_left, product_left, distance_from_spine_left, wrist_distance_left))

                    labels_left = labels_left[1:-1]
                    left_flag = True

            if approach == 'normal':
                if (velocity and right_flag):
                    min_length_right = 0
                else:
                    if velocity:
                        min_length_right = 2
                    else:  #no velocity case
                        min_length_right = 0

                if (velocity and left_flag):
                    min_length_left = 0
                else:
                    if velocity:
                        min_length_left = 2
                    else:  # no velocity case
                        min_length_left = 0


                if (len(data_right) > min_length_right):
                    rp_video_right = rank_pooling(data_right, labels_right, linear=False)
                    np.save((des_dir_name+'/right_inter_new_features.npy'), rp_video_right)
                    #print (des_dir_name+'/right_inter.npy')

                else:
                    total_missing_entries_right += 1
                if (len(data_left) > min_length_left):
                    rp_video_left = rank_pooling(data_left, labels_left, linear=False)
                    np.save((des_dir_name+'/left_inter_new_features.npy'), rp_video_left)
                    #print (des_dir_name+'/left_inter.npy')

                else:
                    total_missing_entries_left += 1


            elif approach == 'symmetric':
                video_data, labels, swap_flag = symmetricity_extraction(data_left, labels_left, data_right,
                                                                        labels_right)
                if swap_flag:
                    swap_count += 1
                rp_video = rank_pooling(video_data, labels, linear=False)
                #np.save((des_dir_name+'/one_arm_inter.npy'), rp_video)

    '''
    if approach == 'normal':
        print 'total_missing_entries_right: ', total_missing_entries_right
        print 'total_missing_entries_left: ', total_missing_entries_left
    elif approach == 'symmetric':
        print 'total_swap count: ', swap_count
    '''


