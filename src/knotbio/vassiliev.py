import os
import csv
import numpy as np
import itertools
import math
from numba import njit

def load(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")

    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"
    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"
    STS = np.loadtxt(os.path.join(master_knots_dir, fname_sts))
    STS = STS.reshape(-1, Nbeads, Nbeads)
    return STS

def combinations(indicies, combinatorics):
    '''
    Generate all possible combinations of indicies (100Cx)
    '''
    return list(itertools.combinations(indicies, combinatorics))

@njit
def vassiliev_combinatorical(STS, test_points):
    '''
    Calculate the Vassiliev invariants for a given knot
    '''
    combinatorics = 4
    vassiliev_data = []
    for idy in range(0, 1):
        integral = 0
        for idx, i in enumerate(test_points):
            if combinatorics == 4:
                integral += abs(STS[idy][i[0], i[2]])*STS[idy][i[1], i[3]] # these are symmetry groups
            elif combinatorics == 6:
                integral += STS[idy][i[0], i[3]]*STS[idy][i[1], i[4]]*STS[idy][i[2], i[5]]
            elif combinatorics == 8:
                integral += STS[idy][i[0], i[4]]*STS[idy][i[1], i[5]]*STS[idy][i[2], i[6]]*STS[idy][i[3], i[7]]

        self_linking = integral / (100 * 100 * 8 * math.pi)
        vassiliev = (6 * self_linking) + (1/4)
        vassiliev_data.append(vassiliev)
    avg_vassiliev = sum(vassiliev_data) / len(vassiliev_data)

    return avg_vassiliev, vassiliev_data

knots = ["0_1", "3_1", "4_1", "5_1", "5_2"]
avgs = []
indicies = np.arange(0, 100, 1)
c = 4
for x in knots:
    STS = load(x, 100, 10) # this is quite slow
    test_points = combinations(indicies, c)
    avg, v_d = vassiliev_combinatorical(STS, test_points)
    with open(f'vassiliev_{x}_comb_{c}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for item in v_d:
            writer.writerow([item])
    avgs.append(avg)

print(f"Combinatorics {4}: ", avgs)
    





