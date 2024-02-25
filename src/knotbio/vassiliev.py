import os
import csv
import numpy as np
import torch
import itertools
import math

def vassiliev_combinatorical(knot_type, Nbeads, pers_len, combinatorics):

    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")
    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"

    STS = np.loadtxt(os.path.join(master_knots_dir, fname_sts))
    torch.tensor(STS, dtype=torch.float32)  
    STS = STS.reshape(-1, 100, 100)

    STA_dat = []
    indicies = np.arange(0, 100, 1)
    test_points = list(itertools.combinations(indicies, combinatorics))
    vassiliev_data = []

    for idy in range(0, 10):
        integral = 0
        for idx, i in enumerate(test_points):
            if combinatorics == 4:
                integral += STS[idy][i[0], i[2]]*STS[idy][i[1], i[3]] # these are symmetry groups
            elif combinatorics == 6:
                integral += STS[idy][i[0], i[3]]*STS[idy][i[1], i[4]]*STS[idy][i[2], i[5]]
            elif combinatorics == 8:
                integral += STS[idy][i[0], i[4]]*STS[idy][i[1], i[5]]*STS[idy][i[2], i[6]]*STS[idy][i[3], i[7]]

        self_linking = integral / (100 * 100 * 8 * math.pi)
        vassiliev = (6 * self_linking) + (1/4)
        vassiliev_data.append(vassiliev)
    avg_vassiliev = sum(vassiliev_data) / len(vassiliev_data)

    with open(f'vassiliev_{knot_type}_combinatorics{combinatorics}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for item in vassiliev_data:
            writer.writerow([item])

    return avg_vassiliev

knots = ["5_1", "7_2"]
avgs = []
for x in knots:
    avg = vassiliev_combinatorical(x, 100, 10, 6)
    avgs.append(avg)

print(avgs)
    





