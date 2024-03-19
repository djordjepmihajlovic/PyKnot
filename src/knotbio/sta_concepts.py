import os
import csv
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plot = False
KnotID = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7"]
KnotIDtex = [r'$5_{1}$', r'$7_{2}$']
peak_order_data = [[],[],[],[],[],[],[]]
peak_separations = [[],[],[],[],[], [], [], []]
sta_area = [[],[],[],[],[], [], [], []]
avg_peak = [[],[], [], [], [], [], []]

for indx, knot in enumerate(KnotID):

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    Nbeads = 100
    pers_len = 10
    dirname = os.path.join(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"))
    fname = os.path.join("SIGWRITHE", f"3DSignedWrithe_{knot}.dat.lp{10}.dat.nos")

    f = open(os.path.join(dirname, fname), "r")

    len_db = 100000
    knot_count = int(len_db/100)
    knot_count = np.arange(0, knot_count, 1)

    sta = [[] for x in knot_count]

    end = 401
    j = 0

    for i in range(0, len_db):
        data = f.readline() 
        if i>=1:
            point = [data]
            point = [float(value) for item in point for value in item.strip().split()]
            sta[j].append(point[2])
            if i % 100 == 0:
                j += 1

    x = np.arange(0, 100, 1)



    prominences = np.linspace(0.75, 0.75, 1)

    # for j in range(0, len(prominences)):

    peak_count_data = [[],[],[],[],[],[],[],[]]

    for i in range(0, len(knot_count)): 

        indices = np.arange(0, len(sta[i]), 1)
        area = np.trapz(y=sta[i], x=indices)
        if knot == "4_1" and i == 0:
            ideal_4_1 = sta[i]

        peaks, properties = find_peaks(sta[i], prominence=0.75)
        vals = properties['prominences']
        prom_sum = np.prod(vals)
        prominence_order = np.array(vals).argsort().tolist()[::-1] # should be invariant to permutation
        prominence_order = [k+1 for k in prominence_order]
        peak_order_data[indx].append(prominence_order)
        peak_count_data[indx].append([len(prominence_order)])
        if len(peaks) > 1:
            sep = np.diff(peaks).tolist()
            extr = 100-peaks[-1]+peaks[0] # looping back
            sep.append(extr)
        else:
            sep = []

        peak_separations[indx].append(sep)
        sta_area[indx].append([area])
    avg_peak[indx].append(np.sum(peak_count_data[indx])/len(peak_count_data[indx]))



    with open(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/knot data/sta concepts/peaks prominence=0.75/peak count/peakcount_{knot}_prom=0.75.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(peak_count_data[indx]) 


if plot == True:

    sns.set_theme(style="white")
    plt.plot(x, sta[0])
    ax = plt.gca()
    plt.ylabel(r'$\omega_{StA}(x_{i})$')
    plt.xlabel(r'Bead index, $x_{i}$')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    for i, idx in enumerate(peaks):
        plt.plot(peaks[i], sta[0][peaks[i]], "x", color = "k")
        if i ==0:
            plt.vlines(peaks[i], ymin=sta[0][peaks[i]] - properties["prominences"][i],
            ymax = sta[0][peaks[i]], color = "k", linestyle = "--", label=f"Peaks")
        else:
            plt.vlines(peaks[i], ymin=sta[0][peaks[i]] - properties["prominences"][i],
                ymax = sta[0][peaks[i]], color = "k", linestyle = "--")
    plt.legend()
    # plt.title(r'$\omega_{StA}$ and peaks at 0.5 prominence.')
    plt.tight_layout()
    plt.show()

    print(avg_peak)

    for indx, x in enumerate(avg_peak):
        # if indx < 5:
        plt.plot(x, prominences, label = f"{KnotIDtex[indx]}")
    plt.legend()
    plt.ylabel("Peak prominence")
    plt.xlabel(r'Avg. $\omega_{StA}$ peak count')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.axhline(y=0.5, color='k', linestyle='--')
    plt.show()

# knot data/sta concepts




   