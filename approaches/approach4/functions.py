import scipy
import matplotlib.pyplot as plt
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    ''' This is the Butterworth bandpass method called by the filter. This is a scipy method'''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # from scipy.signal import firwin
    # b = firwin(order+1, [low, high], pass_zero=False)
    # return b

    b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    ''' This is the Butterworth bandpass filter method. This is a scipy method
        The fs is the sample rate, the lowcut and the hightcut are the  desired cutoff frequencies
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def detrend(means):
    '''  Smoothness prior approach as in the paper appendix:
        "An advanced detrending method with application to HRV analysis "
            by Tarvainen, Ranta-aho and Karjaalainen
        adapted from matlab to python
    '''
    t = len(means)
    l = t / 10  # lambda
    I = np.identity(t)
    D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2],
                            shape=(t - 2, t)).toarray()  # this works the better than the spdiags in python
    mean_stat = (I - np.linalg.inv(I + l ** 2 * (np.transpose(D2).dot(D2)))).dot(means)

    return mean_stat


def preProcess(means, fps):
    '''
        This method will do all the prepocessing of the temporal traces after the zero-mean.
        The steps of pre-processing are explained in the paper defined in the summary of the rppGElaboration function.
    '''

    # 1) detrend with smoothness prior approach (SPA)
    means = detrend(means)

    # 2) butterworth band pass filter (frequencies of 0.7 and 3.5 hz)
    lowcut = 0.7
    highcut = 3.5
    order = 5  # *probably to tune* - the order of the butterworth filter

    means = butter_bandpass_filter(means, lowcut, highcut, fps, order)

    return np.real(means)


def rPPGElaboration(redMeans, greenMeans, blueMeans, fps, show):
    ''' This function will elaborate the ppg signal given the RGB temporal traces, the times of the samples and the fps
        This function is given from the following article:
        "Remote heart rate variability for emotional state monitoring"
            by Y. Benezeth, P. Li, R. Macwan, K. Nakamura, R. Gomez, F. Yang
    '''

    ### Pre- processing
    # 1) the zero mean for the 3 elements
    rawR = redMeans - np.mean(redMeans)
    rawG = greenMeans - np.mean(greenMeans)
    rawB = blueMeans - np.mean(blueMeans)

    # 2) detrend and butterworth bandpass filter
    rawR = preProcess(rawR, fps)
    rawG = preProcess(rawG, fps)
    rawB = preProcess(rawB, fps)

    if show:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(rawR)
        plt.title('processed Red')

        plt.subplot(3, 1, 2)
        plt.plot(rawG)
        plt.title('processed Green')

        plt.subplot(3, 1, 3)
        plt.plot(rawB)
        plt.title('processed Blue')

        plt.show()

    ### rPPG signal S using CHROM method
    # calculation of X
    X = 3 * rawR - 2 * rawG
    # calculation of Y
    Y = (1.5 * rawR) + rawG - (1.5 * rawB)

    ## calculation of alpha
    # calculation of the X's standard deviation (sX)
    sX = np.std(X)

    # calculation of the Y's standard deviation (sY)
    sY = np.std(Y)

    alpha = sX / sY

    # Signal S (rPPG signal)
    S = X - alpha * Y

    return S
