import os
import csv
import numpy as np
import torch
import itertools
import math

def vassiliev(knot_type, Nbeads, pers_len):

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")
    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"

    STS = np.loadtxt(os.path.join(master_knots_dir, fname_sts))
    torch.tensor(STS, dtype=torch.float32)  
    STS = STS.reshape(-1, 100, 100)

    STA_dat = []
    indicies = np.arange(0, 100, 1)
    test_points = list(itertools.combinations(indicies, 4))
    vassiliev_data = []

    for idy in range(0, 10):
        integral = 0
        for idx, i in enumerate(test_points):
            integral += STS[idy][i[0], i[2]]*STS[idy][i[1], i[3]]

        self_linking = integral / (100 * 100 * 8 * math.pi)
        vassiliev = (6 * self_linking) + (1/4)
        vassiliev_data.append(vassiliev)
    avg_vassiliev = sum(vassiliev_data) / len(vassiliev_data)

    with open(f'vassiliev_{knot_type}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for item in vassiliev_data:
            writer.writerow([item])

    return avg_vassiliev

knots = ["0_1", "3_1", "4_1", "5_1", "5_2"]
avgs = []
for x in knots:
    avg = vassiliev(x, 100, 10)
    avgs.append(avg)

print(avgs)
    





