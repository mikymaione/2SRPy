import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from .jade import jadeR


def approach1(images, show, fps):
    face_red_means = list()
    face_green_means = list()
    face_blue_means = list()

    min_hz = .75  # 45 BPM
    max_hz = 4  # 240 BPM

    face_cascade = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))

    faces = face_cascade.detectMultiScale(images[0])
    facesize = (faces[0][2], faces[0][3])

    for i in range(len(images)):
        frame = images[i]

        faces = face_cascade.detectMultiScale(frame, minSize=facesize, maxSize=facesize)

        view = np.array(frame)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                reduced_w = int(w * .6)

                x = x + (int((w - reduced_w) / 2))
                w = reduced_w

                face = view[y:y + h, x:x + w]

                face_blue_means.append(np.mean(face[:, :, 0]))
                face_green_means.append(np.mean(face[:, :, 1]))
                face_red_means.append(np.mean(face[:, :, 2]))

        elif len(faces) == 0:
            print("No face detected...")
            return 0

        if show:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    mu_r = np.mean(face_red_means)
    mu_g = np.mean(face_green_means)
    mu_b = np.mean(face_blue_means)

    std_r = np.std(face_red_means)
    std_g = np.std(face_green_means)
    std_b = np.std(face_blue_means)

    face_red_means = (face_red_means - mu_r) / std_r
    face_green_means = (face_green_means - mu_g) / std_g
    face_blue_means = (face_blue_means - mu_b) / std_b

    if show:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(face_red_means)
        plt.title('RED')

        plt.subplot(3, 1, 2)
        plt.plot(face_green_means)
        plt.title('GREEN')

        plt.subplot(3, 1, 3)
        plt.plot(face_blue_means)
        plt.title('BLUE')

        plt.show()

    X = np.vstack([face_red_means, face_green_means, face_blue_means])
    unmixing_mat = np.asarray(jadeR(X, verbose=False))
    y = np.dot(unmixing_mat, X)

    if show:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(y[0, :])
        plt.title('1st (JADE)')

        plt.subplot(3, 1, 2)
        plt.plot(y[1, :])
        plt.title('2nd (JADE)')

        plt.subplot(3, 1, 3)
        plt.plot(y[2, :])
        plt.title('3rd (JADE)')

        plt.show()

    # Get second component
    second = y[1, :]

    # Compute FFT
    fft = np.abs(np.fft.rfft(second))

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