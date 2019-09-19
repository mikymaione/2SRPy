import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.color
from . import evm
from scipy import signal


def butterworth_filter(data, low, high, sample_rate, order=3):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate

    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


def extractRegionValue(filtered, area, show):
    # area: 0 - forehead
    #       1 - cheeks
    #       2 - neck
    #       3 - all

    result_fhead = list()
    result_cheek = list()
    result_neck = list()

    for i in range(len(filtered)):
        frame = filtered[i]
        startw, starth, _ = frame.shape

        if area == 0 or area == 3:  # forehead
            x_f = int(startw * .34)
            y_f = int(starth * .11)
            w_f = int(startw * .32)
            h_f = int(starth * .12)

            roi = frame[y_f:y_f + h_f, x_f:x_f + w_f]
            roi = np.mean(skimage.color.rgb2yiq(roi)[:, :, 0])
            result_fhead.append(roi)

        if area == 1 or area == 3:  # cheeks

            # left cheek
            x_c1 = int(startw * .22)
            y_c = int(starth * .54)
            w_c = int(startw * .14)
            h_c = int(starth * .11)

            roi = frame[y_c:y_c + h_c, x_c1:x_c1 + w_c]
            roi1 = np.mean(skimage.color.rgb2yiq(roi)[:, :, 0])

            # right cheek
            x_c2 = int(startw * .64)

            roi = frame[y_c:y_c + h_c, x_c2:x_c2 + w_c]
            roi = np.mean(skimage.color.rgb2yiq(roi)[:, :, 0])

            roi += roi1
            result_cheek.append(roi)

        if area == 2 or area == 3:  # neck
            x_n = int(startw * .60)
            y_n = int(starth * .90)
            w_n = int(startw * .10)
            h_n = int(starth * .09)

            roi = frame[y_n:y_n + h_n, x_n:x_n +
                                           w_n]
            roi = np.mean(skimage.color.rgb2yiq(roi)[:, :, 0])
            result_neck.append(roi)

        if show:
            cv2.rectangle(frame, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 255, 0), 2)
            cv2.rectangle(frame, (x_c1, y_c), (x_c1 + w_c, y_c + h_c), (0, 255, 0), 2)
            cv2.rectangle(frame, (x_c2, y_c), (x_c2 + w_c, y_c + h_c), (0, 255, 0), 2)
            cv2.rectangle(frame, (x_n, y_n), (x_n + w_n, y_n + h_n), (0, 255, 0), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    return result_fhead, result_cheek, result_neck


def approach3(images, show, fps):
    face_frames = list()

    face_cascade = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))

    min_hz = 0.83  # 50 BPM
    max_hz = 2.9  # 174 BPM
    alpha = 100
    num_levels = 6

    faces = face_cascade.detectMultiScale(images[0])
    facesize = (faces[0][2], faces[0][3])

    for i in range(len(images)):
        frame = images[i]

        view = np.array(frame)

        faces = face_cascade.detectMultiScale(frame, minSize=facesize, maxSize=facesize)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = view[y:y + h, x:x + w]

                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # separate luminance and chrominance
                face = skimage.color.rgb2yiq(face)

                face_frames.append(face)

        else:
            print("No face detected...")
            return 0

    filtered = evm.main(face_frames, fps, alpha, num_levels)

    # extract ROIs
    forehead, cheeks, neck = extractRegionValue(filtered, 3, show)

    data = np.array([forehead, cheeks, neck])

    ydata = np.mean(data, axis=0)

    ydata = (ydata - np.mean(ydata)) / np.std(ydata)

    ydata = butterworth_filter(ydata, min_hz, max_hz, fps, order=3)

    if show:
        plt.plot(ydata)
        plt.title('Y-component VPG total mean ROI')
        plt.show()

    # Compute FFT
    fft = np.abs(np.fft.rfft(ydata))

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
