# MIT License
#
# Copyright (c) 2019 Michele Maione, mikymaione@hotmail.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import sys
import matplotlib.pyplot as plt

from utility.evaluateAlgorithm import evaluate

try:
    setN = sys.argv[3]
    ok_videos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] #set3
    #ok_videos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] #set0
    algos = [0, 1, 2, 3, 4, 5]
    nomi_algos = ["5", "4", "3", "2", "1", "2SR"]
    tot = [0, 0, 0, 0, 0, 0]
    tot_num_of_Videos = len(ok_videos)
    show = False
    TuttiInsieme = False
    writeLatex = True
    writeCSV = True
    RMSEv = []
    outMancanti = []

    for algo in algos:
        RMSEs = []

        for ok_video in ok_videos:
            data = "./samples/{}/{}/data.csv".format(ok_video, setN)
            out = "outs{}/out_{}_{}.txt".format(setN, ok_video, algo)
            RMSE = 100

            try:
                RMSE = evaluate(data, out, sys.argv[1], sys.argv[2])
            except Exception as error2:
                if ok_video not in outMancanti:
                    outMancanti.append(ok_video)

                print('Error on {}: {}'.format(ok_video, error2))

            RMSEs.append(RMSE)

        if TuttiInsieme:
            plt.plot(RMSEs, label="Approccio " + nomi_algos.pop())

        RMSEv.append(RMSEs)

    if show:
        if TuttiInsieme:
            plt.legend()
            plt.show()

        if not TuttiInsieme:
            nomi_algos.pop()
            a0 = RMSEv.pop()

            while len(RMSEv) > 0:
                an = RMSEv.pop()
                plt.plot(a0, label="Approccio 2SR")
                plt.plot(an, label="Approccio " + nomi_algos.pop())
                plt.legend()
                plt.show()


    if writeLatex:
        latex = ""
        # 1 &17.30 &26.25 &27.15 &15.75 &16.20 &5.93 \\ \hline

        num_video = -1
        for ok_video in ok_videos:
            num_video += 1
            latex += "{}".format(ok_video)

            lower_el = 100
            for a in RMSEv:
                el = a[num_video]

                if el < lower_el:
                    lower_el = el

            u = -1
            for a in RMSEv:
                u += 1
                el = a[num_video]
                tot[u] += el

                if lower_el == el:
                    bold1 = "\\textbf{"
                    bold2 = "}"
                else:
                    bold1 = ""
                    bold2 = ""

                latex += " &{}{:.2f}{}".format(bold1, el, bold2)

            latex += " \\\ \hline\n"

        latex += "AVG"
        for t in tot:
            latex += " &{:.2f}".format(t / tot_num_of_Videos)

        latex += " \\\ \hline\n"

        fout = open("latex_{}.tex".format(setN), 'w')
        fout.write(latex)
        fout.close()

    if writeCSV:
        fout = open("valutazione_{}.csv".format(setN), 'w')

        num_video = -1
        for ok_video in ok_videos:
            num_video += 1
            CSV = "{};".format(ok_video)

            for a in RMSEv:
                CSV += "{:.2f};".format(a[num_video])

            fout.write(CSV + '\n')

        fout.close()

except Exception as error:
    print('Error: ')
    print(error)
