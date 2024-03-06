import os
import csv
import numpy as np
from numba import njit
from multiprocessing import Pool

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
def compute_integral(idy, STS):
    integral1 = 0
    integral2 = 0
    integral3 = 0
    integral4 = 0
    integral5 = 0
    integral6 = 0
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
                                        for o in range(0, N):
                                            for p in range(0, N):
                                                if n<o and o<p:
                                                    integral1 += STS[idy][i, m]*STS[idy][j, o]*STS[idy][k, n]*STS[idy][l, p]
                                                    integral2 += STS[idy][i, n]*STS[idy][j, l]*STS[idy][k, p]*STS[idy][m, o]
                                                    integral3 += STS[idy][i, k]*STS[idy][j, l]*STS[idy][m, o]*STS[idy][n, p]

                                                    integral4 += STS[idy][i, p]*STS[idy][j, l]*STS[idy][k, n]*STS[idy][m, o]
                                                    integral5 += STS[idy][i, j]*STS[idy][k, m]*STS[idy][l, n]*STS[idy][o, p]
                                                    integral6 += STS[idy][i, j]*STS[idy][k, l]*STS[idy][m, n]*STS[idy][o, p]

    return (integral1, integral2, integral3, integral4, integral5, integral6)

def chord_combinatorical_STS_parallel(STS):
    '''
    Calculate the chord diagram integral for a given knot
    '''
    samples = 1000
    pool = Pool(10) # run 10 processes
    results = [pool.apply_async(compute_integral, args=(idy, STS)) for idy in range(samples)]
    pool.close()
    pool.join()

    chord_data = [[] for _ in range(6)]
    for result in results:
        integrals = result.get()
        for i in range(6):
            chord_data[i].append(integrals[i] / (100 * 100 * 100 * 100))

    return chord_data

def main():
    knots = ["3_1"]
    avgs = []
    for x in knots:
        STS = load_STS(x, 100, 10)
        print("StS loaded")
        print("Calculating A4 chord diagrams...")
        chord_data = chord_combinatorical_STS_parallel(STS)

        for i, data in enumerate(chord_data):
            with open(f'{x}_A4_{i+1}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([[item] for item in data])

main()

