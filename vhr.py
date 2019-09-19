import sys
import cv2
import skimage
import numpy as np
import approaches.approach1.approach1 as a1
import approaches.approach2.approach2 as a2
import approaches.approach3.approach3 as a3
import approaches.approach4.approach4 as a4
import approaches.approach5.approach5 as a5


class VHR:

    def __init__(self, inVid, m, w, s, outfile, v=0):
        self.inputVideo = inVid;
        self.method = m;
        self.window = w;
        self.step = s;
        self.out = outfile;
        self.show = v;

    def doCalc(self):

        methods = {
            1: a1.approach1,
            2: a2.approach2,
            3: a3.approach3,
            4: a4.approach4,
            5: a5.approach5,
        }

        # Read video
        video = cv2.VideoCapture(self.inputVideo)

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        # Calculate the number of frames per interval
        fps = int(video.get(cv2.CAP_PROP_FPS))
        totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        listImages = []
        listBpms = []

        ret, frame = video.read()

        while ret:
            listImages.append(frame)
            ret, frame = video.read()

        video.release()

        data = np.arange(0, totalFrames)
        indexes = skimage.util.view_as_windows(data, fps * self.window, step=fps * self.step)

        for i in range(len(indexes)):
            print("Processing %s window %d/%d" % (self.inputVideo, i + 1, len(indexes)))
            curImages = listImages[indexes[i][0]: indexes[i][-1]]

            # Apply method X to current list of images
            bpm = methods[self.method](curImages, self.show, fps)

            listBpms.append(bpm)

        fout = open(self.out, 'w')

        for x in listBpms:
            fout.write(str(x) + '\n')

        fout.close()
