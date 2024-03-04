import os
import csv
import numpy as np
import itertools
import math
from numba import njit
from argparse import ArgumentParser

def load_STS(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")

    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"
    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    STS = np.loadtxt(os.path.join(master_knots_dir, fname_sts))
    STS = STS.reshape(-1, Nbeads, Nbeads)
    return STS

@njit
def arf_STS(STS):
    '''
    Calculate the Arf invariants for a given knot
    '''
    samples = 100
    arf_data = []

    for idy in range(0, samples): # samples
        integral = 0
        N = 100
        for i in range(0, (N/2)):
            integral += STS[idy][(2*i - 1), (2*i - 1)] * STS[idy][(2*i), (2*i)]

        arf = (integral%2) / (100 * 100)
        arf_data.append(arf)


    avg_arf = sum(arf_data) / len(arf_data)

    return arf_data, avg_arf

def main():
    knots = ["0_1", "3_1", "4_1", "5_1", "5_2"]
    avgs = []
    for x in knots:
        STS = load_STS(x, 100, 10) # this is quite slow
        print("StS loaded")
        print("Calculating Vassiliev invariants...")
        arf, av = arf_STS(STS)

        with open(f'arf_{x}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in arf:
                writer.writerow([item])
        avgs.append(av)


    print(f"Arf invariants: ", avgs)

main()

