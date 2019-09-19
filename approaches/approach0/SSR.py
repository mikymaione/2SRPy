# MIT License
#
# Copyright (c) 2019 Michele Maione, mikymaione@hotmail.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import cv2
import numpy

from approaches.approach0.SkinColorFilter import SkinColorFilter


class SSR():
    """
    This class implements the Spatial Subspace Rotation for Remote Photoplethysmography

    It is based on the work published in "A Novel Algorithm for Remote Photoplethysmography - Spatial Subspace Rotation",
    Wenjin Wang, Sander Stuijk, and Gerard de Haan, IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 63, NO. 9, SEPTEMBER 2016
    """

    def calulate_pulse_signal(self, images, show, fps):
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
        k : int
            The number of frame elaborated

        P: numpy.ndarray | dim: K = len(images)
            The pulse signal
        """

        k = 0  # the number of frame elaborated
        K = len(images)  # the number of frame to elaborate
        l = fps  # The temporal stride to use

        # the pulse signal
        P = numpy.zeros(K)  # 1 | dim: K

        # store the eigenvalues Λ and the eigenvectors U at each frame
        Λ = numpy.zeros((3, K), dtype='float64')  # dim: 3xK
        U = numpy.zeros((3, 3, K), dtype='float64')  # dim: 3x3xK

        # object detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.
        haarcascade_frontalface_default_xml_file = os.path.join('data', 'haarcascade_frontalface_default.xml')
        haar_face_cascade_classifier = cv2.CascadeClassifier(haarcascade_frontalface_default_xml_file)

        # a class to perform skin color filtering based on the work published in "Adaptive skin segmentation via feature-based face detection".
        skin_filter = SkinColorFilter()
        minNeighbors = 3  # OpenCV default

        for i in range(K):  # 2
            img = images[i]  # dim: HxWx3

            # detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
            n_faces = 100
            while n_faces > 1:
                faces = haar_face_cascade_classifier.detectMultiScale(img, minNeighbors=minNeighbors)  # dim: 1x4
                n_faces = len(faces)

                if n_faces == 0:
                    if minNeighbors == 3:  # no face detected
                        return 0, P
                    else:
                        minNeighbors = 3  # OpenCV default
                elif n_faces == 1:
                    # get: skin pixels
                    V = self.__get_skin_pixels(skin_filter, faces, img, show, k == 0)  # 3 | dim: (W×H)x3

                    # build the correlation matrix
                    C = self.__build_correlation_matrix(V)  # 3 | dim: 3x3

                    # get: eigenvalues Λ, eigenvectors U
                    Λ[:, k], U[:, :, k] = self.__eigs(C)  # 4 | dim Λ: 3 | dim U: 3x3

                    # build p and add it to the pulse signal P
                    if k >= l:  # 5
                        τ = k - l  # 5
                        p = self.__build_p(τ, k, l, U, Λ)  # 6, 7, 8, 9, 10, 11 | dim: l
                        P[τ:k] += p  # 11

                    k = k + 1
                else:
                    minNeighbors += 5
                    if minNeighbors > 40:  # third cycle
                        return 0, P

        return k, P

    def __build_correlation_matrix(self, V):
        # V dim: (W×H)x3
        V_T = V.T  # dim: 3x(W×H)

        N = V.shape[0]

        # build the correlation matrix
        C = numpy.dot(V_T, V)  # dim: 3x3
        C = C / N

        return C

    def __get_skin_pixels(self, skin_filter, faces, img, show, do_skininit):
        """
        get eigenvalues and eigenvectors, sort them.

        Parameters
        ----------
        C: numpy.ndarray
            The RGB values of skin-colored pixels.

        Returns
        -------
        Λ: numpy.ndarray
            The eigenvalues of the correlation matrix

        U: numpy.ndarray
            The (sorted) eigenvectors of the correlation matrix
        """
        for (x, y, w, h) in faces:
            ROI = img[y:y + h, x:x + w]  # dim: wxhx3

        if do_skininit:
            skin_filter.estimate_gaussian_parameters(ROI)

        skin_mask = skin_filter.get_skin_mask(ROI)  # dim: wxh

        V = ROI[skin_mask]  # dim: (w×h)x3
        V = V.astype('float64') / 255.0

        if show:
            # show the skin in the image along with the mask
            cv2.imshow("Image", numpy.hstack([img]))
            cv2.imshow("ROI", numpy.hstack([ROI]))

            if cv2.waitKey(20) & 0xFF == ord('q'):
                return

        return V

    def __eigs(self, C):
        """
        get eigenvalues and eigenvectors, sort them.

        Parameters
        ----------
        C: numpy.ndarray
            The RGB values of skin-colored pixels.

        Returns
        -------
        Λ: numpy.ndarray
            The eigenvalues of the correlation matrix

        U: numpy.ndarray
            The (sorted) eigenvectors of the correlation matrix
        """

        # get eigenvectors and sort them according to eigenvalues (largest first)
        Λ, U = numpy.linalg.eig(C)  # dim Λ: 3 | dim U: 3x3

        idx = Λ.argsort()  # dim: 3x1
        idx = idx[::-1]  # dim: 1x3

        Λ_ = Λ[idx]  # dim: 3
        U_ = U[:, idx]  # dim: 3x3

        return Λ_, U_

    def __build_p(self, τ, k, l, U, Λ):
        """
        builds P

        Parameters
        ----------
        k: int
            The frame index

        l: int
            The temporal stride to use

        U: numpy.ndarray
            The eigenvectors of the c matrix (for all frames up to counter).

        Λ: numpy.ndarray
            The eigenvalues of the c matrix (for all frames up to counter).

        Returns
        -------
        p: numpy.ndarray
            The p signal to add to the pulse.
        """

        # SR'
        SR = numpy.zeros((3, l), 'float64')  # dim: 3xl
        z = 0

        for t in range(τ, k, 1):  # 6, 7
            a = Λ[0, t]
            b = Λ[1, τ]
            c = Λ[2, τ]
            d = U[:, 0, t].T
            e = U[:, 1, τ]
            f = U[:, 2, τ]
            g = U[:, 1, τ].T
            h = U[:, 2, τ].T

            x1 = a / b
            x2 = a / c
            x3 = numpy.outer(e, g)
            x4 = numpy.dot(d, x3)
            x5 = numpy.outer(f, h)
            x6 = numpy.dot(d, x5)
            x7 = numpy.sqrt(x1)
            x8 = numpy.sqrt(x2)
            x9 = x7 * x4
            x10 = x8 * x6
            x11 = x9 + x10

            SR[:, z] = x11  # 8 | dim: 3
            z += 1

        # build p and add it to the final pulse signal
        s0 = SR[0, :]  # dim: l
        s1 = SR[1, :]  # dim: l

        p = s0 - ((numpy.std(s0) / numpy.std(s1)) * s1)  # 10 | dim: l
        p = p - numpy.mean(p)  # 11

        return p  # dim: l
