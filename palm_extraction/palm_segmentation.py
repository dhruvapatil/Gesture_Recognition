import numpy as np
import os
import skimage.io
import matplotlib.pyplot as plt
from skimage import measure
from skimage.util import pad
import sys

def get_all_frames(dir_path,depth=True):
    num_frames = len(os.listdir(dir_path))
    depth_images = []
    for frame in range(1, num_frames + 1):
        image_path = os.path.join(dir_path, "%d.jpg" % frame)
        image = skimage.io.imread(image_path, depth)

        depth_images.append(image)

    depth_images = np.stack(depth_images)

    return depth_images

def depth_segment():

    palm_location_file = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/palm_location/valid/%03d/K_%05d.npy"
    image_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/%03d/K_%05d"
    hand_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/hands/individual/valid/%03d/K_%05d"

    for video in range(5600,5600):
        folder = ((video - 1) / 200) + 1

        hand_dir = hand_root % (folder, video)
        try:
            os.makedirs(hand_dir)
        except:
            pass

        palm_location = np.load(palm_location_file % (folder, video))
        depth_images = get_all_frames(image_root % (folder, video))
        print folder, video, palm_location.shape, depth_images.shape, hand_dir

        image_name = ["%d_right.png", "%d_left.png"]

        for frame, (palm, image) in enumerate(zip(palm_location, depth_images)):



            for i in range(2):

                if np.sum(palm[i]) != -2:

                    x_start = palm[i, 0] - 50
                    x_end = x_start + 100

                    y_start = palm[i, 1] - 50
                    y_end = y_start + 100

                    width = [[0, 0], [0, 0]]

                    #if y_end<=320:
                    #    continue
                    print frame,


                    if x_end > 240:
                        width[0][1] = x_end - 240
                        x_end = 240

                    if y_end > 320:
                        width[1][1] = y_end - 320
                        y_end = 320

                    if x_start < 0:
                        width[0][0] = -x_start
                        x_start = 0

                    if y_start < 0:
                        width[1][0] = -y_start
                        y_start = 0


                    palm_depth = image[palm[i, 0], palm[i, 1]]
                    translate = 0.5 - palm_depth

                    hand = np.copy(image[x_start:x_end, y_start:y_end]) + translate
                    hand = pad(hand, width, 'constant', constant_values=1)


                    palm_depth_max = 0.5 + 0.1
                    palm_depth_min = 0.5 - 0.2

                    hand[hand > palm_depth_max] = 1
                    hand[hand < palm_depth_min] = 1

                    binary_hand = np.copy(hand)
                    binary_hand[binary_hand != 1] = 2

                    labels = measure.label(binary_hand)
                    palm_label = labels[50, 50]
                    labels[labels != palm_label] = 0
                    hand[labels == 0] = 1
                else:
                    hand = np.ones((100, 100))

                hand *= 255
                hand = hand.astype("uint8")

                skimage.io.imsave(os.path.join(hand_dir, image_name[i] % frame), hand)

                # plt.show()
        print


def depth(folder):
    palm_location_file = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/palm_location/train/%03d/K_%05d.npy"
    image_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/train/%03d/K_%05d"
    hand_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/hands/depth/individual/train/%03d/K_%05d"


    start = (folder-1)*200+1
    end = start+200
    print start, end
    for video in range(start,end):
        hand_dir = hand_root % (folder, video)
        try:
            os.makedirs(hand_dir)
        except:
            pass

        palm_location = np.load(palm_location_file % (folder, video))
        depth_images = get_all_frames(image_root % (folder, video))
        print folder, video, palm_location.shape, depth_images.shape, hand_dir

        image_name = ["%d_right.png", "%d_left.png"]

        for frame, (palm, image) in enumerate(zip(palm_location, depth_images)):

            for i in range(2):

                if np.sum(palm[i]) != -2:

                    x_start = palm[i, 0] - 55
                    x_end = x_start + 110

                    y_start = palm[i, 1] - 55
                    y_end = y_start + 110

                    width = [[0, 0], [0, 0]]


                    #print frame,

                    if x_end > 240:
                        width[0][1] = x_end - 240
                        x_end = 240

                    if y_end > 320:
                        width[1][1] = y_end - 320
                        y_end = 320

                    if x_start < 0:
                        width[0][0] = -x_start
                        x_start = 0

                    if y_start < 0:
                        width[1][0] = -y_start
                        y_start = 0



                    hand = np.copy(image[x_start:x_end, y_start:y_end])
                    hand = pad(hand, width, 'constant', constant_values=1)



                    hand *= 255
                    hand = hand.astype("uint8")

                    skimage.io.imsave(os.path.join(hand_dir, image_name[i] % frame), hand)
                    #plt.imshow(hand,cmap="gray")
                    #plt.show()
        print


def rgb(folder):
    palm_location_file = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/palm_location/valid/%03d/K_%05d.npy"
    image_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/valid/%03d/M_%05d"
    hand_root = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/hands/rgb/individual/valid/%03d/M_%05d"

    start = (folder - 1) * 200 + 1
    end = start + 200
    print start, end
    for video in range(start, end):

        hand_dir = hand_root % (folder, video)
        try:
            os.makedirs(hand_dir)
        except:
            pass

        palm_location = np.load(palm_location_file % (folder, video))
        depth_images = get_all_frames(image_root % (folder, video),False)
        print folder, video, palm_location.shape, depth_images.shape, hand_dir

        image_name = ["%d_right.png", "%d_left.png"]

        for frame, (palm, image) in enumerate(zip(palm_location, depth_images)):

            for i in range(2):

                if np.sum(palm[i]) != -2:
                    x_start = palm[i, 0] - 55
                    x_end = x_start + 110

                    y_start = palm[i, 1] - 55
                    y_end = y_start + 110

                    width = [[0, 0], [0, 0],[0,0]]

                    #print frame,

                    if x_end > 240:
                        width[0][1] = x_end - 240
                        x_end = 240

                    if y_end > 320:
                        width[1][1] = y_end - 320
                        y_end = 320

                    if x_start < 0:
                        width[0][0] = -x_start
                        x_start = 0

                    if y_start < 0:
                        width[1][0] = -y_start
                        y_start = 0



                    hand = np.copy(image[x_start:x_end, y_start:y_end])
                    hand = pad(hand, width, 'constant', constant_values=255)



                    hand = hand.astype("uint8")

                    skimage.io.imsave(os.path.join(hand_dir, image_name[i] % frame), hand)
                    #plt.imshow(hand)
                    #plt.show()

        print

#depth(int(sys.argv[1]))
rgb(int(sys.argv[1]))