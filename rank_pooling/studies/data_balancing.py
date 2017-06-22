import numpy as np
import os
from os.path import join
from collections import Counter

main_folder = '../index_log_files'
#features_train = np.load(join(main_folder, 'features_train.npy'))
labels_train = np.load(join(main_folder, 'labels_train.npy'))

#features_val = np.load(join(main_folder, 'features_val.npy'))
#labels_val = np.load(join(main_folder, 'labels_val.npy'))

labels = map(int,labels_train)

print Counter(labels)