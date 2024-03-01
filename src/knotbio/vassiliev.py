import os
import csv
import numpy as np
import itertools
import math
from numba import njit
from argparse import ArgumentParser

def load(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")

    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"
    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"
    STS = np.loadtxt(os.path.join(my_knot_dir, fname_sts))
    STS = STS.reshape(-1, Nbeads, Nbeads)
    return STS

def combinations(indicies, combinatorics):
    '''
    Generate all possible combinations of indicies (100Cx)
    '''
    return list(itertools.combinations(indicies, combinatorics))

@njit
def vassiliev_combinatorical(STS, test_points, combinatorics, t):
    '''
    Calculate the Vassiliev invariants for a given knot
    '''
    samples = 3
    vassiliev_data = []
    c = 6
    for idy in range(0, samples): # samples
        integral = 0
        for idx, i in enumerate(test_points):
            print(idx)

            if combinatorics == 4:
                integral += STS[idy][i[0], i[2]]*STS[idy][i[1], i[3]] # these are symmetry groups

            elif combinatorics == 6:
                integral += STS[idy][i[0], i[3]]*STS[idy][i[1], i[4]]*STS[idy][i[2], i[5]]                                                 

        if combinatorics == 4:
            self_linking = integral / (100 * 100 * 8 * math.pi)
            vassiliev = (6 * self_linking) + (1/4)
            vassiliev_data.append(vassiliev)

        elif combinatorics == 6:
            self_linking = integral / (100 * 100 * 100)
            vassiliev_data.append(self_linking)

    avg_vassiliev = sum(vassiliev_data) / len(vassiliev_data)

    return avg_vassiliev, vassiliev_data

def main():
    knots = ["3_1"]
    avgs = []
    indicies = np.arange(0, 100, 1)
    for x in knots:
        STS = load(x, 100, 10) # this is quite slow
        print("StS loaded")
        test_points = combinations(indicies, 6)
        print("Combinations generated")
        print("Calculating Vassiliev invariants...")
        avg, v_d = vassiliev_combinatorical(STS, test_points, 6, 1)
        with open(f'vassiliev_{x}_comb_{6}_{1}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in v_d:
                writer.writerow([item])
        avgs.append(avg)

    print(f"Combinatorics {6}: ", avgs)

main()

# if __name__ == "__main__":

#     par = ArgumentParser()
#     par.add_argument("-c", "--combination", type=int, default=4, help="Combinations N s.t., 100C(N)")
#     par.add_argument("-t", "--type", type=int, default=1, help="Combinatorial set (arbitrary)")
#     args = par.parse_args()

#     c = args.combination
#     t = args.type

#     main()
    
# broken now and not sure why





