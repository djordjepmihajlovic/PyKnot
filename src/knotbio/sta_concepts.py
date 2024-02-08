import os
import csv
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

knot = "0_1"

dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/"
fname = os.path.join("SIGWRITHE", f"3DSignedWrithe_{knot}.dat.lp{10}.dat.nos")

f = open(os.path.join(dirname, fname), "r")

len_db = 100000
knot_count = int(len_db/100)
knot_count = np.arange(0, knot_count, 1)

sta = [[] for x in knot_count]

end = 401
j = 0

print(sta)

for i in range(0, len_db):
    data = f.readline() 
    if i>=1:
        point = [data]
        point = [float(value) for item in point for value in item.strip().split()]
        sta[j].append(point[2])
        if i % 100 == 0:
            j += 1

x = np.arange(0, 100, 1)

peak_order_data = []
peak_count_data = []
peak_separations = []

for i in range(0, len(knot_count)): 

    peaks, properties = find_peaks(sta[i], prominence=0.5)
    vals = properties['prominences']
    prominence_order = np.array(vals).argsort().tolist()[::-1] # should be invariant to permutation
    prominence_order = [k+1 for k in prominence_order]
    peak_order_data.append(prominence_order)
    peak_count_data.append(len(prominence_order))
    if len(peaks) > 1:
        sep = np.diff(peaks).tolist()
        extr = 100-peaks[-1]+peaks[0] # looping back
        sep.append(extr)
    else:
        sep = []

    peak_separations.append(sep)


print(peak_separations)

with open(f'../../knot data/sta concepts/peaks prominence=0.5/peak order/peakpermute_{knot}_prom=0.5.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(peak_order_data) 

# sns.set_theme(style="darkgrid")
# plt.plot(x, sta[0])
# ax = plt.gca()
# ax.get_yaxis().set_visible(False)
# plt.title(f"StA writhe")
# plt.tight_layout()
# for i, idx in enumerate(peaks):
#     plt.plot(peaks[i], sta[0][peaks[i]], "x")
#     plt.vlines(peaks[i], ymin=sta[0][peaks[i]] - properties["prominences"][i],
#            ymax = sta[0][peaks[i]], color = "C1", linestyle = "dotted")
    
# plt.show()

# knot data/sta concepts
