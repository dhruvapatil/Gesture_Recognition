import os
import numpy as np
import skimage.io
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

file_name = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/heat_maps/valid/%03d/K_%05d/heat_map.npy"
image_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/%03d/K_%05d"
dest_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/palm_images/%03d/K_%05d"

def get_all_frames(dir_path):
    
    num_frames = len(os.listdir(dir_path))
    depth_images = []
    color_images = []
    for frame in range(1, num_frames+1):
        image_path = os.path.join(dir_path, "%d.jpg"%frame) 
        image = skimage.io.imread(image_path,True)



        color_image = skimage.io.imread(image_path.replace("K","M"))

        depth_images.append(image)
        color_images.append(color_image)


    depth_images = np.stack(depth_images)
    color_images = np.stack(color_images)

    return depth_images, color_images

def find_peaks(heat_map):
    map_left = np.zeros(heat_map.shape)
    map_left[1:, :] = heat_map[:-1, :]
    map_right = np.zeros(heat_map.shape)
    map_right[:-1, :] = heat_map[1:, :]
    map_up = np.zeros(heat_map.shape)
    map_up[:, 1:] = heat_map[:, :-1]
    map_down = np.zeros(heat_map.shape)
    map_down[:, :-1] = heat_map[:, 1:]

    peaks_binary = np.logical_and.reduce((heat_map >= map_left, heat_map >= map_right, heat_map >= map_up, heat_map >= map_down, heat_map >= 0.05))
    peaks = np.nonzero(peaks_binary)
    if len(peaks[0])>1:
        max_index = np.argmax(heat_map[peaks])
        peaks = (peaks[0][max_index:max_index+1], peaks[1][max_index:max_index+1])

    if len(peaks[0]) ==0 :
        peaks = (np.array([-1]),np.array([-1]))

    return peaks


def get_peaks(heat):
    peak_list = []
    for part in range(2):
        heat_map = heat[:, :, part]
        smoothed_map = gaussian_filter(heat_map, sigma=3)
        peaks = find_peaks(smoothed_map)

        #print peaks, np.hstack(list(peaks))
        peak_list.append(np.hstack(list(peaks)))

    return peak_list

def plot_palms(depth, rgb, heat):
    plt.clf()
    plt.subplot(231)
    plt.imshow(rgb)

    plt.subplot(232)
    plt.imshow(depth, cmap="gray")

    plt.subplot(233)
    plt.imshow(depth, cmap="gray")

    plt.subplot(234)
    plt.imshow(depth, cmap="gray")
    plt.imshow(heat[:, :, 0], alpha=0.5)
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(depth, cmap="gray")
    plt.imshow(heat[:, :, 1], alpha=0.5)
    plt.colorbar()

    colors = ['r', 'b']

    peak_list = get_peaks(heat)
    for part in range(2):
        peaks = peak_list[part]

        #print "Peaks ", peaks
        plt.subplot(233)
        if len(peaks) > 0:
            plt.plot(peaks[1], peaks[0], 'o', color=colors[part])
        plt.xlim([0, 320])
        plt.ylim([240, 0])
    return peak_list


def save_all_hands():
    dest_dir = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/palm_location/valid/%03d"
    dest = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/palm_location/valid/%03d/K_%05d.npy"
    for i in range(1,5785):
        folder = ((i-1)/200)+1

        try:
            os.makedirs(dest_dir%folder)
        except:
            pass

        heat_maps = np.load(file_name % (folder, i))
        heat_maps[heat_maps > 1] = 1

        palms = []
        for index, heat in enumerate(heat_maps):
            heat = resize(heat, (240, 320, 2), order=3)

            palms.append(get_peaks(heat))
        palms = np.array(palms)
        print folder, i,palms.shape
        #print palms
        np.save(dest%(folder,i), palms)



def visualize():
    for i in range(9003,18000):
        folder = ((i-1)/200)+1

        depth_images, color_images = get_all_frames(image_root%(folder,i))
        heat_maps = np.load(file_name%(folder,i))
        heat_maps[heat_maps>1] = 1
        print folder, i, heat_maps.shape, depth_images.shape, color_images.shape

        dest = dest_root%(folder, i)
        try:
            os.makedirs(dest)
        except:
            pass

        palms = []
        for index,(depth, rgb, heat) in enumerate(zip(depth_images, color_images, heat_maps)):

            print index,
            heat = resize(heat, (240, 320, 2), order=3)
            peak_list = plot_palms(depth, rgb, heat)
            print peak_list
            palms.append(peak_list)

            plt.savefig(os.path.join(dest,"%d.jpg"%(index+1)))
        print
        print np.array(palms).shape


if __name__ == "__main__":
    save_all_hands()