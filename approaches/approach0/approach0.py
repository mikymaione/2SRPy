# MIT License
#
# Copyright (c) 2019 Michele Maione, mikymaione@hotmail.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from approaches.BPM_Converter import component_to_bpm
from approaches.approach0.SSR import SSR


def approach0(images, show, fps):
    """
    Parameters
    ----------
    images: List<numpy.ndarray | dim: HxWx3>
        The images to elaborate

    show: int [0/1]
        Show the plot

    fps: int
        Frame per seconds

    Returns
    -------
    bpm: float64
        Beat per minute
    """

    min_hz = 0.83  # 50 BPM
    max_hz = 3.33  # 200 BPM

    # the pulse signal P
    ssr = SSR()
    k, P = ssr.calulate_pulse_signal(images, show, fps)  # dim: K

    return component_to_bpm(min_hz, max_hz, fps, k, P, show)
