import os
import csv
import numpy as np
from numba import njit

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
def chord_combinatorical_STS(STS):
    '''
    Calculate the chord diagram integral for a given knot
    '''
    samples = 1000
    chord_data_1 = []
    chord_data_2 = []
    chord_data_3 = []

    for idy in range(0, samples): # samples
        integral1 = 0
        integral2 = 0
        integral3 = 0
        N = 100
        for i in range(0, N):
            for j in range(0, N):
                if i<j:
                    for k in range(0, N):
                        for l in range(0, N):
                            if j<k and k<l:
                                for m in range(0, N):
                                    for n in range(0, N):
                                        if l<m and m<n:
                                            integral1 += STS[idy][i, j]*STS[idy][k, l]*STS[idy][m, n] 
                                            integral2 += STS[idy][i, m]*STS[idy][j, n]*STS[idy][k, l]
                                            integral3 += STS[idy][i, k]*STS[idy][j, m]*STS[idy][l, n]

        chord_data_1.append(integral1 / (100 * 100 * 100))
        chord_data_2.append(integral2 / (100 * 100 * 100))
        chord_data_3.append(integral3 / (100 * 100 * 100))

    return chord_data_1, chord_data_2, chord_data_3


def main():
    knots = ["0_1", "3_1", "4_1", "5_1", "5_2"]
    avgs = []
    for x in knots:
        STS = load_STS(x, 100, 10) # this is quite slow
        print("StS loaded")
        print("Calculating A3 chord diagrams...")
        chord_1, chord_2, chord_3 = chord_combinatorical_STS(STS)

        with open(f'{x}_A3_1.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in chord_1:
                writer.writerow([item])

        with open(f'{x}_A3_2.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in chord_2:
                writer.writerow([item])

        with open(f'{x}_A3_3.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in chord_3:
                writer.writerow([item])


