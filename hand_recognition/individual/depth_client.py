import numpy as np
from realtime_hand_recognition import RealTimeHandRecognition
import skimage.io
import matplotlib.pyplot as plt
import os
from collections import Counter
from collections import Counter
import cPickle


def get_all_frames(dir_path, hand):
    if hand == "RH":
        image_pattern = "%d_left.png"
    else:
        image_pattern = "%d_right.png"

    frames = os.listdir(dir_path)

    frames = list(set([int(f.split("_")[0]) for f in frames]))
    frames.sort()

    depth_images = []
    frame_list = []
    for frame in frames:
        image_path = os.path.join(dir_path, image_pattern % frame)
        try:
            image = skimage.io.imread(image_path, True)
        except:
            continue

        if np.sum(image) == 2550000:
            continue
        image = np.array(image, "float32")
        image /= 255
        image -= 0.5
        image *= 2
        depth_images.append(image)
        frame_list.append(frame)

    try:
        depth_images = np.stack(depth_images)[:, :, :, np.newaxis]
    except:
        return None, None

    return depth_images, frame_list


def get_gt_labels():
    label_file = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/train_list.txt"
    label_dict = {}
    for line in open(label_file).readlines():
        _, path, label = line.strip().split(" ")
        path = path.split("_")[-1].replace(".avi", "")
        label_dict[int(path)] = int(label) - 1
    return label_dict


def train(hand):
    label_dict = get_gt_labels()
    num_gestures = 250
    print hand, num_gestures

    hand_classfier = RealTimeHandRecognition(hand, num_gestures)

    gt_list = []
    pred_list = []
    prob_dict = {}

    tp = 0.0
    fp = 0.0
    for video in range(1, 30001):
        folder = ((video - 1) / 200) + 1
        dir_path = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/hands/depth_segmented/individual/train/%03d/K_%05d/" % (
        folder, video)

        depth_images, frame_list = get_all_frames(dir_path, hand)
        most_common = 1
        if depth_images is not None:
            print folder, video, depth_images.shape,
            probs = hand_classfier.classify(depth_images)
            pred = np.argmax(probs, axis=1)

            gt_list += [label_dict[video]] * len(pred)
            pred_list += list(pred)

            most_common, num_most_common = Counter(pred).most_common(1)[0]
            print most_common, label_dict[video], np.max(probs[:, most_common])

            prob_dict[video] = [probs, frame_list]

        if most_common == label_dict[video]:
            tp += 1
        else:
            fp += 1

    print len(gt_list), len(pred_list)

    ftp = 0.0
    for g, p in zip(gt_list, pred_list):
        if g == p:
            ftp += 1

    print hand, ftp, len(gt_list), ftp / len(gt_list)
    print hand, tp, tp + fp, tp / (tp + fp)

    with open("/s/red/a/nobackup/cwc/skeleton/ChaLearn17/predictions/depth_segmented/train_%s.p" % hand, "wb") as fd:
        cPickle.dump(prob_dict, fd)


def eval():
    label_dict = get_gt_labels()
    hand = "RH"
    num_gestures = 250
    print hand, num_gestures

    hand_classfier = RealTimeHandRecognition(hand, num_gestures)

    gt_list = []
    pred_list = []
    prob_dict = {}

    tp = 0.0
    fp = 0.0
    for video in range(30001, 35878):
        folder = ((video - 1) / 200) + 1
        dir_path = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/hands/depth_segmented/individual/train/%03d/K_%05d/" % (
        folder, video)

        depth_images, frame_list = get_all_frames(dir_path, hand)
        most_common = 1
        if depth_images is not None:
            print folder, video, depth_images.shape,
            probs = hand_classfier.classify(depth_images)
            pred = np.argmax(probs, axis=1)

            gt_list += [label_dict[video]] * len(pred)
            pred_list += list(pred)

            most_common, num_most_common = Counter(pred).most_common(1)[0]
            print most_common, label_dict[video], np.max(probs[:, most_common])

            # print pred, label_dict[video]

            prob_dict[video] = [probs, frame_list]

        if most_common == label_dict[video]:
            tp += 1
        else:
            fp += 1

    print len(gt_list), len(pred_list)

    ftp = 0.0
    for g, p in zip(gt_list, pred_list):
        if g == p:
            ftp += 1

    print hand, ftp, len(gt_list), ftp / len(gt_list)
    print hand, tp, tp + fp, tp / (tp + fp)

    with open("/s/red/a/nobackup/cwc/skeleton/ChaLearn17/predictions/depth_segmented/eval_%s.p" % hand, "wb") as fd:
        cPickle.dump(prob_dict, fd)


def valid(hand):
    num_gestures = 250
    print hand, num_gestures

    hand_classfier = RealTimeHandRecognition(hand, num_gestures)
    prob_dict = {}
    for video in range(1, 5785):
        folder = ((video - 1) / 200) + 1
        dir_path = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/hands/depth_segmented/individual/valid/%03d/K_%05d/" % (
        folder, video)
        depth_images, frame_list = get_all_frames(dir_path, hand)

        if depth_images is not None:
            print folder, video, depth_images.shape
            probs = hand_classfier.classify(depth_images)
            prob_dict[video] = [probs, frame_list]

    with open("/s/red/a/nobackup/cwc/skeleton/ChaLearn17/predictions/depth_segmented/valid_%s.p" % hand, "wb") as fd:
        cPickle.dump(prob_dict, fd)


def features(hand):
    num_gestures = 250
    print hand, num_gestures

    hand_classfier = RealTimeHandRecognition(hand, num_gestures)

    for video in range(1, 5800):
        folder = ((video - 1) / 200) + 1
        dir_path = "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/frames/hands/depth_segmented/individual/valid/%03d/K_%05d/" % (
        folder, video)
        depth_images, frame_list = get_all_frames(dir_path, hand)

        if depth_images is not None:
            features = hand_classfier.features(depth_images)
            print folder, video, depth_images.shape, features.shape
            np.save(
                "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/features/valid/%03d/K_%05d/%s_depth_segmented_features.npy" % (
                folder, video, hand), features)
            np.save(
                "/s/red/a/nobackup/cwc/skeleton/ChaLearn17/features/valid/%03d/K_%05d/%s_depth_segmented_frames.npy" % (
                folder, video, hand), frame_list)


if __name__ == '__main__':
    features("LH")
    # train("LH")
    # valid("RH")
    # eval()
