import numpy as np
import os
from os.path import join

'''
src_dir = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/rankpooling/val/'

folder_list = os.listdir(src_dir)

#train_folders = folder_list[:150]
#val_folders = folder_list[150:]
#print len(train_folders), len(val_folders)

sub_folder = 'val'
features = []
filenames = []
rp_length_threshold = 4

folders = folder_list
print len(folders)

for folder in folders:
    videos_list = os.listdir(join(src_dir, folder))

    for video in videos_list:
        files = os.listdir(join(src_dir, folder, video))

        if 'one_arm.npy' in files:
            feature_arm = np.load(join(src_dir, folder, video, 'one_arm.npy'))
            if len(feature_arm) == 4:
                features.append(feature_arm)
                filenames.append(join(sub_folder, folder, video))

        
        if 'left.npy' in files and 'right.npy' in files:
            left_feature = np.load(join(src_dir, folder, video, 'left.npy'))
            right_feature = np.load(join(src_dir, folder, video, 'right.npy'))
            if (len(left_feature) == len(right_feature) == 4):
                features.append(np.hstack((left_feature, right_feature)))
                filenames.append(join(sub_folder, folder, video))
        elif 'left.npy' in files and 'right.npy' not in files:
            left_feature = np.load(join(src_dir, folder, video, 'left.npy'))
            if (len(left_feature) == 4):
                right_feature = np.zeros(4)
                features.append(np.hstack((left_feature, right_feature)))
                filenames.append(join(sub_folder, folder, video))
        elif 'right.npy' in files and 'left.npy' not in files:
            right_feature = np.load(join(src_dir, folder, video, 'right.npy'))
            if (len(right_feature) == 4):
                left_feature = np.zeros(4)
                features.append(np.hstack((left_feature, right_feature)))
                filenames.append(join(sub_folder, folder, video))
        

print len(features), len(filenames)
features_and_filenames = zip(features, filenames)
features_and_filenames = [j for j in features_and_filenames if len(j[0])==rp_length_threshold]

print len(features_and_filenames)
print features_and_filenames[0]

#np.save('../valid_both_arms.npy', features_and_filenames)
'''

def generate_features_and_filenames(src_dir, des_filename, approach, category, velocity_flag = False):
    sub_folder = src_dir.split('/')[-1]
    print 'subfolder is: ', sub_folder

    folder_list = os.listdir(src_dir)
    if category == 'train':
        folders = folder_list[:150]
    elif category == 'val':
        folders = folder_list[150:]
    elif category == 'valid':
        folders = folder_list



    if approach == 'symmetric':
        if velocity_flag:
            rp_length_threshold = 8
        else:
            rp_length_threshold = 4
    elif approach == 'combined':
        if velocity_flag:
            rp_length_threshold = 16
        else:
            rp_length_threshold = 8

    features = []
    filenames = []


    for folder in folders:
        videos_list = os.listdir(join(src_dir, folder))

        for video in videos_list:
            files = os.listdir(join(src_dir, folder, video))
            if approach == 'symmetric':
                if 'one_arm.npy' in files:
                    feature_arm = np.load(join(src_dir, folder, video, 'one_arm.npy'))
                    if len(feature_arm) == 4:
                        features.append(feature_arm)
                        filenames.append(join(sub_folder, folder, video))

            elif approach == 'combined':
                if 'left_inter.npy' in files and 'right_inter.npy' in files:
                    left_feature = np.load(join(src_dir, folder, video, 'left_inter.npy'))
                    right_feature = np.load(join(src_dir, folder, video, 'right_inter.npy'))
                    if (len(left_feature) == len(right_feature) == 8):
                        features.append(np.hstack((left_feature, right_feature)))
                        filenames.append(join(sub_folder, folder, video))
                elif 'left_inter.npy' in files and 'right_inter.npy' not in files:
                    left_feature = np.load(join(src_dir, folder, video, 'left_inter.npy'))
                    if (len(left_feature) == 8):
                        right_feature = np.zeros(8)
                        features.append(np.hstack((left_feature, right_feature)))
                        filenames.append(join(sub_folder, folder, video))
                elif 'right_inter.npy' in files and 'left_inter.npy' not in files:
                    right_feature = np.load(join(src_dir, folder, video, 'right_inter.npy'))
                    if (len(right_feature) == 8):
                        left_feature = np.zeros(8)
                        features.append(np.hstack((left_feature, right_feature)))
                        filenames.append(join(sub_folder, folder, video))


    print len(features), len(filenames)
    features_and_filenames = zip(features, filenames)
    features_and_filenames = [j for j in features_and_filenames if len(j[0]) == rp_length_threshold]

    print len(features_and_filenames)
    print features_and_filenames[0]

    np.save(des_filename, features_and_filenames)



def generate_features_and_labels(file_name_parent, category, des_filename_features, des_filename_labels):
    features = np.load(file_name_parent)
    print features.shape

    rp_features = []
    rp_labels = []

    if category == 'train' or category == 'val':
        file_name = '../files/train_list.txt'
        des_filename = '../files/labels_file.txt'
        for j in features:
            for line in open(des_filename):
                line = line.rstrip().split()
                if (j[1] == line[0]):
                    # print j[1], j[0], line[1]
                    rp_features.append(j[0])
                    rp_labels.append(line[1])

        rp_features = np.vstack(rp_features)
        np.save(des_filename_features, rp_features)
        np.save(des_filename_labels, rp_labels)
        print rp_features.shape, len(rp_labels)

    elif category == 'valid':  # here the order is not taken into account, the order is specified in the features_and_filenames.npy file
        for j in features:
            rp_features.append(j[0])

        rp_features = np.vstack(rp_features)
        np.save(des_filename_features, rp_features)
        print rp_features.shape

