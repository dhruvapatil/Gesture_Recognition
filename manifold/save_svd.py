import numpy as np
import os
from TensorUnfold import UnfoldCube
import scipy.linalg as LA
import sys

rgb_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/data_cube_max_diff/valid/%03d/K_%05d"
depth_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/data_cube_max_diff/valid/%03d/M_%05d"
dest_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/manifold/svd_max_diff/valid/%03d/M_%05d"

file_pattern_list = ["%s_w.npy", "%s_h.npy", "%s_t.npy"]


def create_data_cubes(video):
    folder = ((video - 1) / 200) + 1
    print folder, video
    for root in [rgb_root, depth_root]:
        file_list = os.listdir(root % (folder, video))
        dest_dir = root % (folder, video)
        dest_dir = dest_dir.replace("data_cube_max_diff", "svd_max_diff")

        try:
            os.makedirs(dest_dir)
        except:
            pass

        for file_name in file_list:
            cube = np.load(os.path.join(root % (folder, video), file_name))
            for mode in range(1, 4):
                unfolded_cube = UnfoldCube(cube, mode)
                unfolded_cube -= np.mean(unfolded_cube, axis=0)
                basis, _, _ = LA.svd(unfolded_cube.T, full_matrices=False)
                np.save(os.path.join(dest_dir, file_pattern_list[mode - 1] % file_name.replace(".npy", "")), basis)


create_data_cubes(int(sys.argv[1]))
