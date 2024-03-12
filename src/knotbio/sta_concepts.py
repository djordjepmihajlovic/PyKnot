import os
import csv
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

knot = "0_1"
KnotID = ["0_1", "3_1", "4_1", "5_1", "5_2"]
KnotIDtex = [r'$0_{1}$', r'$3_{1}$', r'$4_{1}$', r'$5_{1}$', r'$5_{2}$']
peak_order_data = [[],[],[],[],[]]
peak_separations = [[],[],[],[],[]]
sta_area = [[],[],[],[],[]]
avg_peak = [[],[],[],[],[]]

for indx, knot in enumerate(KnotID):

    dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/"
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



    prominences = np.linspace(0.05, 3.5, 50)


    for j in range(0, len(prominences)):

        peak_count_data = [[],[],[],[],[],[],[],[]]

        for i in range(0, len(knot_count)): 



            indices = np.arange(0, len(sta[i]), 1)
            area = np.trapz(y=sta[i], x=indices)
            if knot == "4_1" and i == 0:
                ideal_4_1 = sta[i]

            peaks, properties = find_peaks(sta[i], prominence=prominences[j])
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



    # with open(f'../../knot data/sta concepts/peaks prominence=variable/peak count/peakcount_{knot}_prom=var.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(peak_count_data[indx]) 

sns.set_theme(style="darkgrid")
plt.plot(x, ideal_4_1)
ax = plt.gca()
ax.get_yaxis().set_visible(False)
plt.title(f"StA writhe")
plt.tight_layout()
plt.show()
# for i, idx in enumerate(peaks):
#     plt.plot(peaks[i], sta[0][peaks[i]], "x")
#     plt.vlines(peaks[i], ymin=sta[0][peaks[i]] - properties["prominences"][i],
#            ymax = sta[0][peaks[i]], color = "C1", linestyle = "dotted")
# print(avg_peak)

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




   