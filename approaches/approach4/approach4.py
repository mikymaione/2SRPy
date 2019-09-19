import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from . import functions


def getSkinPixels(roi, show):
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    converted = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(roi, roi, mask=skinMask)

    if show:
        # show the skin in the image along with the mask
        cv2.imshow("images", np.hstack([roi, skin]))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            return

    return skin.T


def approach4(images, show, fps):
    ## Variables
    face_red_means = list()  # the lists of the forehead means
    face_green_means = list()
    face_blue_means = list()
    # bpms = list()  # a series with the second in question and the bpm calculated for that second
    min_hz = 0.7
    max_hz = 3.5

    face_cascade = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))

    # first loop: it takes the frames, saves the means and works on them. no display of the video, just a . every second
    for i in range(len(images)):
        frame = images[i]
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:  # if the detection is at least 1 face
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                [b, g, r] = getSkinPixels(face, show)

                face_red_means.append(np.mean(r))
                face_green_means.append(np.mean(g))
                face_blue_means.append(np.mean(b))
        else:
            print("Too many faces...")
            return 0

    S = functions.rPPGElaboration(face_red_means, face_green_means, face_blue_means, fps, show)

    # Compute FFT
    fft = np.abs(np.fft.rfft(S))

    # Generate list of frequencies that correspond to the FFT values
    freqs = fps / len(images) * np.arange(len(fft))
    bpm = 0
    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
    # because they correspond to impossible BPM values.
    while True:
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < min_hz or bps > max_hz:
            fft[max_idx] = 0
        else:
            bpm = bps * 60.0
            break

    if show:
        plt.plot(freqs, fft)
        plt.plot(freqs[max_idx], fft[max_idx], label='argmax', marker="o", ls="", ms=3)  # the peaks of the ppg
        plt.text(freqs[max_idx], fft[max_idx], ' Peak corresponding to Maximum freq')
        plt.show()

    print('[BPM] %d' % bpm)

    return bpm
