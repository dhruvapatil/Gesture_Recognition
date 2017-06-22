import os
import cv2
from interfaces.constants import *

def save_frames(video_path, dest_dir):
    video_fd = cv2.VideoCapture(video_path)
    count = 1
    while True:
        ret, image = video_fd.read()
        if image is None:
            break
        file_name = os.path.join(dest_dir,"%d.jpg"%count)
        cv2.imwrite(file_name, image)
        count += 1
    print video_path, count



def save_frames_all():
    for root, dir_list, file_list in os.walk(video_root):
        if dir_list == []:
            print root
            for video_name in file_list:
                video_path = os.path.join(root, video_name)
                print video_path
                dest_dir = os.path.join(frames_root, video_name).replace(".avi","")
                print dest_dir
                try:
                    os.makedirs(dest_dir)
                except:
                    pass

                save_frames(video_path, dest_dir)

if __name__ == "__main__":
    save_frames_all()