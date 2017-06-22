import numpy as np


#Spine-Base Normalization
def normalize_spine_base(data):
    b = np.copy(data)
    spine = np.zeros(3)
    for i in range(3):
        spine[i] = np.mean(b[:, i])
    m, n = b.shape

    for i in range(n):
        b[:, i] -= spine[i % 3]

    return b


def normalize_spine_base_dataset(data_list):
    return [normalize_spine_base(data) for data in data_list]


#Normalization by other joints
def normalize_by_joint(data, joint_index, verbose = False):
    dims = 2
    #print 'normalization joint index is: ', joint_index
    b = np.copy(data)
    b = b.astype(np.float64)
    joint = np.zeros(dims)

    for i in range(dims):
        joint[i] = np.mean(b[:, (joint_index*dims + i)])

    m, n = b.shape
    for i in range(n):
        b[:, i] -= joint[i % dims]

    '''
    if verbose:
        print 'data before normalization: ', data
        print 'mean of the spine joint:', joint
        print 'data after normalization: ', b
    '''
    return b



def normalize_joint_dataset(data_list, joint_index):
    return [normalize_by_joint(data, joint_index) for data in data_list]

