import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA


def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = signal.firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False, window=window, scale=False)
    return taps


# Gets the region of interest for the forehead.
def get_forehead_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    l_eye = int(points[36, 0]) + abs(int(points[39, 0]) - int(points[36, 0])) / 2
    r_eye = int(points[42, 0]) + abs(int(points[45, 0]) - int(points[42, 0])) / 2

    fourd = abs(r_eye - l_eye)
    d = fourd / 4

    bottom = int(points[39, 1]) - d
    top = bottom - (3 * d) / 2
    left = l_eye + 0.5 * d
    right = left + 3 * d

    return int(left), int(right), int(top), int(bottom)


def approach2(images, show, fps):
    face_red_means = list()
    face_green_means = list()
    face_blue_means = list()

    min_hz = .5  # 30 bpm
    max_hz = 3.7  # 222 bpm
    ntaps = 32

    detector = dlib.get_frontal_face_detector()
    # Predictor pre-trained model can be downloaded from:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor(os.path.join('data', 'shape_predictor_68_face_landmarks.dat'))

    for i in range(len(images)):
        frame = images[i]
        # Detect face using dlib
        faces = detector(frame, 0)

        view = np.array(frame)

        if len(faces) > 0:
            face_points = predictor(frame, faces[0])

            # Get the regions of interest.
            fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)

            fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]

            face_blue_means.append(np.sum(fh_roi[:, :, 0]))
            face_green_means.append(np.sum(fh_roi[:, :, 1]))
            face_red_means.append(np.sum(fh_roi[:, :, 2]))

        else:
            print("No face detected...")
            return 0

        if show:
            # Draw green rectangles around our regions of interest (ROI)
            cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
            cv2.imshow('frame', view)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    if show:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(face_red_means)
        plt.title('RED raw')

        plt.subplot(3, 1, 2)
        plt.plot(face_green_means)
        plt.title('GREEN raw')

        plt.subplot(3, 1, 3)
        plt.plot(face_blue_means)
        plt.title('BLUE raw')

        plt.show()

    b = bandpass_firwin(ntaps, min_hz, max_hz, fps)

    fb_filt = signal.lfilter(b, [1.0], face_blue_means)
    fr_filt = signal.lfilter(b, [1.0], face_red_means)
    fg_filt = signal.lfilter(b, [1.0], face_green_means)

    if show:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(fr_filt)
        plt.title('RED filtered')

        plt.subplot(3, 1, 2)
        plt.plot(fg_filt)
        plt.title('GREEN filtered')

        plt.subplot(3, 1, 3)
        plt.plot(fb_filt)
        plt.title('BLUE filtered')

        plt.show()

    X = np.vstack([fr_filt, fg_filt, fb_filt])

    y = PCA(n_components=3).fit_transform(X.T)

    if show:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(y[:, 0])
        plt.title('1st (PCA)')

        plt.subplot(3, 1, 2)
        plt.plot(y[:, 1])
        plt.title('2nd (PCA)')

        plt.subplot(3, 1, 3)
        plt.plot(y[:, 2])
        plt.title('3nd (PCA)')

        plt.show()

    # Get second component
    second = y[:, 1]

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
