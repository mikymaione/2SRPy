# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:34:44 2018

@author: conte
"""
import sys
import cv2
import skimage
import numpy
import approaches.approach0.approach0 as a0
import approaches.approach1.approach1 as a1
import approaches.approach2.approach2 as a2
import approaches.approach3.approach3 as a3
import approaches.approach4.approach4 as a4
import approaches.approach5.approach5 as a5

if len(sys.argv) != 7:
    usg = 'Usage : virtualHeartRate <video file> <method (int)> <X range (in sec)> <X step (in sec)> <out file (txt)> <show>'
    print(usg)
    # print('Implemented methods (see README) : 1 - approach 1 ; 2 - approach 2; 3 - approach 3 ; 5 - approach 5 ')
    exit()

videofilename = sys.argv[1]  # video file
methodNumber = int(sys.argv[2])  # method (int)
intX = int(sys.argv[3])  # X range (in sec)
step = int(sys.argv[4])  # X step (in sec)
outfilename = sys.argv[5]  # out file (txt)
show = int(sys.argv[6])  # show

# Read video
video = cv2.VideoCapture(videofilename)

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Calculate the number of frames per interval
fps = int(video.get(cv2.CAP_PROP_FPS))
totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

listImages = []
listBpms = []

while True:
    ret, frame = video.read()

    if not ret:
        break

    listImages.append(frame)

data = numpy.arange(0, totalFrames)
indexes = skimage.util.view_as_windows(data, fps * intX, step=fps * step)

try:
    for i in range(len(indexes)):
        print("Processing window %d/%d" % (i + 1, len(indexes)))

        curImages = listImages[indexes[i][0]: indexes[i][-1]]

        bpm = 0
        # Apply method X to current list of images
        if methodNumber == 0:
            bpm = a0.approach0(curImages, show, fps)
        elif methodNumber == 1:
            bpm = a1.approach1(curImages, show, fps)
        elif methodNumber == 2:
            bpm = a2.approach2(curImages, show, fps)
        elif methodNumber == 3:
            bpm = a3.approach3(curImages, show, fps)
        elif methodNumber == 4:
            bpm = a4.approach4(curImages, show, fps)
        elif methodNumber == 5:
            bpm = a5.approach5(curImages, show, fps)

        listBpms.append(bpm)
except Exception as error:
    print('Error: ')
    print(error)

fout = open(outfilename, 'w')

for x in listBpms:
    fout.write(str(x) + '\n')

fout.close()
video.release()
