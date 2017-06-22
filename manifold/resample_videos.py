import os
import skimage.io
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
import math
import sys

def pad_video(image_stack, pad_frames):
    pad_start = pad_frames/2
    pad_end = pad_frames- pad_start

    start_stack = np.repeat(image_stack[:1,:,:],pad_start,axis=0)
    end_stack = np.repeat(image_stack[-1:, :, :], pad_end, axis=0)

    return np.vstack([start_stack,image_stack,end_stack])

def sample_video(image_stack, sample_factor):
    return image_stack[::sample_factor,:,:]


def create_data_cubes(video):
    folder = ((video - 1) / 200) + 1
    window_length = 25
    print folder, video
    for d in ["/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/%03d/K_%05d/", "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/%03d/M_%05d/"]:
        dir_path = d % (folder, video)
        num_frames = len(os.listdir(dir_path))

        dest_dir = dir_path.replace("frames","manifold/data_cube_max_diff")

        try:
            os.makedirs(dest_dir)
        except:
            pass

        image_stack = [skimage.transform.pyramid_reduce(skimage.io.imread(os.path.join(dir_path, "%d.jpg" % f), True), 8) for f in range(1, num_frames + 1)]
        image_stack = np.stack(image_stack)


        if num_frames<window_length+1:
            pad_frames = window_length+1-num_frames
            image_stack = pad_video(image_stack, pad_frames)

        if num_frames>(window_length+1)*2:
            resample_factor = num_frames/(window_length+1)
            image_stack = sample_video(image_stack, resample_factor)


        image_stack = np.abs(np.diff(image_stack, axis=0))
        #print "After diff ",image_stack.shape
        num_frames = image_stack.shape[0]

        if "K" in d:
            diff_list = []
            for i in range(0,num_frames-25+1):
                cube = image_stack[i:i+25,:,:]
                #np.save(os.path.join(dest_dir,"%d.npy"%(i)), cube)
                diff_list.append(np.sum(cube))

            max_cube_index = np.argmax(diff_list)
        max_cube = image_stack[max_cube_index:max_cube_index+25,:,:]
        np.save(os.path.join(dest_dir, "%d.npy" % (0)), max_cube)

if __name__ == "__main__":
    create_data_cubes(int(sys.argv[1]))
