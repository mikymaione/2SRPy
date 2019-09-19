# MIT License
#
# Copyright (c) 2019 Michele Maione, mikymaione@hotmail.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy


def component_to_bpm(min_hz, max_hz, fps, len_images, component, show):
    bpm = 0
    max_idx_prev = -1

    # Compute FFT
    fft = numpy.abs(numpy.fft.rfft(component))

    # Generate list of frequencies that correspond to the FFT values
    freqs = fps / len_images * numpy.arange(len(fft))

    # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ] because they correspond to impossible BPM values.
    while True:
        max_idx = fft.argmax()

        if max_idx_prev == max_idx:
            break

        max_idx_prev = max_idx

        bps = freqs[max_idx]

        if bps < min_hz or bps > max_hz:
            fft[max_idx] = 0
        else:
            bpm = bps * 60.0
            break

    if show:
        import matplotlib.pyplot as plt
        plt.plot(freqs, fft)
        plt.plot(freqs[max_idx], fft[max_idx], label='argmax', marker="o", ls="", ms=3)  # the peaks of the ppg
        plt.text(freqs[max_idx], fft[max_idx], ' Peak corresponding to Maximum freq')
        plt.show()

    print('[BPM] %d' % bpm)

    return bpm
