import os
import csv
import seaborn as sns
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plot = True
KnotID = ["3_1"]
KnotIDtex = [r'$5_{1}$', r'$7_{2}$']
peak_order_data = [[] for _ in KnotID]
peak_separations = [[] for _ in KnotID]
sta_area = [[] for _ in KnotID]
avg_peak = [[] for _ in KnotID]

def detect_peaks(image):
    neighborhood = generate_binary_structure(2,2)   

    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image == 0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    detected_peaks = local_max ^ eroded_background
    return detected_peaks

for indx, knot in enumerate(KnotID):

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    Nbeads = 100
    pers_len = 10
    dirname = os.path.join(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"))
    fname = os.path.join("SIGWRITHEMATRIX", f"3DSignedWritheMatrix_{knot}.dat.lp{pers_len}.dat")

    len_db = 100000
    knot_count = int(len_db/100)
    knot_count = np.arange(0, knot_count, 1)

    STS = np.loadtxt(os.path.join(dirname, fname))
    STS = STS.reshape(-1, Nbeads, Nbeads)

    peak_count_data = [[] for _ in KnotID]


    # for i in range(0, 1): 
    #     detect_peaks(STS[i])
    #     peak_count_data[indx].append(len(detect_peaks(STS[i])))


    # with open(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/knot data/sts concepts/peaks/peakcount_{knot}.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(peak_count_data[indx]) 

    # with open(f'/storage/cmstore02/groups/TAPLab/djordje_mlknots/PyKnot/knot data/sta concepts/area/area_{knot}.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(sta_area[indx]) 

if plot == True:

    sns.set_theme(style="white")
    sample = STS[0]
    sample_peaks = detect_peaks(sample)
    plt.subplot(1, 2, 1)
    plt.imshow(sample, cmap='viridis')
    plt.subplot(1, 2, 2)
    plt.imshow(sample_peaks, cmap='viridis')
    plt.tight_layout()
    plt.savefig('peaks StS.png')
    print(sample_peaks.sum())

# knot data/sta concepts