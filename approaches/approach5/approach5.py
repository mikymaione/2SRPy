import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Constants
min_hz = 0.83  # 50 BPM
max_hz = 3.33  # 200 BPM


# Creates the specified Butterworth filter and applies it.
# See:  http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')

    return signal.lfilter(b, a, data)


# Gets the region of interest for the forehead.
def get_forehead_roi(face_points):
    # Store the points in a Numpy array so we can easily get the min and max for x and y via slicing
    points = np.zeros((len(face_points.parts()), 2))

    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Forehead area between eyebrows
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98

    return int(left), int(right), int(top), int(bottom)


# Gets the region of interest for the nose.
def get_nose_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))

    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Nose and cheeks
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(points[36, 0])
    min_y = int(points[28, 1])
    max_x = int(points[45, 0])
    max_y = int(points[33, 1])
    left = min_x
    right = max_x
    top = min_y + (min_y * 0.02)
    bottom = max_y + (max_y * 0.02)

    return int(left), int(right), int(top), int(bottom)


# Gets region of interest that includes forehead, eyes, and nose.
# Note:  Combination of forehead and nose performs better.  This is probably because this ROI includes
#	     the eyes, and eye blinking adds noise.
def get_full_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))

    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # Only keep the points that correspond to the internal features of the face (e.g. mouth, nose, eyes, brows).
    # The points outlining the jaw are discarded.
    # See:  https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    min_x = int(np.min(points[17:47, 0]))
    min_y = int(np.min(points[17:47, 1]))
    max_x = int(np.max(points[17:47, 0]))
    max_y = int(np.max(points[17:47, 1]))

    center_x = min_x + (max_x - min_x) / 2
    # center_y = min_y + (max_y - min_y) / 2
    left = min_x + int((center_x - min_x) * 0.15)
    right = max_x - int((max_x - center_x) * 0.15)
    top = int(min_y * 0.88)
    bottom = max_y

    return int(left), int(right), int(top), int(bottom)


def sliding_window_demean(signal, num_windows):
    window_size = int(round(len(signal) / num_windows))
    demeaned = np.zeros(signal.shape)

    for i in range(0, len(signal), window_size):
        if i + window_size > len(signal):
            window_size = len(signal) - i

        slice = signal[i:i + window_size]

        if slice.size == 0:
            print('Empty Slice: size={0}, i={1}, window_size={2}'.format(signal.size, i, window_size))
            print(slice)

        demeaned[i:i + window_size] = slice - np.mean(slice)

    return demeaned


def get_avg(roi1, roi2):
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0

    if np.isnan(avg) or np.isinf(avg):
        return 50
    else:
        return avg


def approach5(images, show, fps):
    # Lists for storing video frame data
    allvalues = []

    detector = dlib.get_frontal_face_detector()
    # Predictor pre-trained model can be downloaded from:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor(os.path.join('data', 'shape_predictor_68_face_landmarks.dat'))

    for i in range(len(images)):
        frame = images[i]

        # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
        # The original frame will be used to compute heart rate.
        view = np.array(frame)

        # Detect face using dlib
        faces = detector(frame, 0)

        if len(faces) > 0:
            face_points = predictor(frame, faces[0])

            # Get the regions of interest.
            fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
            nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

            if show:
                # Draw green rectangles around our regions of interest (ROI)
                cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
                cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2)

                cv2.imshow('frame', view)

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

            # Slice out the regions of interest (ROI) and average them
            fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
            nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
            avg = get_avg(fh_roi, nose_roi)

            # Add value and time to lists
            allvalues.append(avg)

        else:
            print("No face detected...")
            return 0

    # Smooth the signal by detrending and demeaning
    detrended = signal.detrend(np.array(allvalues), type='linear')
    demeaned = sliding_window_demean(detrended, 15)

    # Filter signal with Butterworth bandpass filter
    filtered = butterworth_filter(demeaned, min_hz, max_hz, fps, order=5)

    if show:
        plt.plot(filtered)
        plt.title("Filtered data")
        plt.show()

    # Compute FFT
    fft = np.abs(np.fft.rfft(filtered))

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
