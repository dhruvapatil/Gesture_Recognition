import os
import numpy as np
from os.path import join
from communication.Contact import send_email
from dataset.Extraction import extract_rank_pooling_features_inter
from dataset.Preprocessing import (generate_features_and_filenames, generate_features_and_labels)


if __name__ == "__main__":

    main_folder = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/skeleton/inter_skeletons_valid'
    des_folder = '/s/red/a/nobackup/cwc/skeleton/ChaLearn17/rankpooling/val'


    import sys
    import socket

    hostname = socket.gethostname()
    subject_start = 'Processing for ' + hostname + ' has started'
    subject_end = 'Processing for ' + hostname + ' is complete'

    list_index =  int(sys.argv[1])

    #message_start = '\nlist_index: ' + str(list_index) + '\nMain folder Path: ' + main_folder + '\nDesPath: ' + des_folder
    #send_email('patil.dhruva@gmail.com', subject_start, message_start)

    extract_rank_pooling_features_inter(main_folder, des_folder, list_index, category='valid', velocity=True, approach='normal')

    #message_end = hostname + ' has finished processing  list_index ' + str(list_index)
    #send_email('patil.dhruva@gmail.com', subject_end, message_end)


    #generate_features_and_filenames(des_folder, '../index_log_files/inter_valid_feature_and_filename.npy', approach='combined', category='valid', velocity_flag=True)
    #generate_features_and_labels('../index_log_files/inter_valid_feature_and_filename.npy', 'valid', '../index_log_files/features_valid.npy', '../index_log_files/labels_valid.npy')