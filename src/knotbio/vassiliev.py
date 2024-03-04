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

def load_STA(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")

    fname_sta = f"SIGWRITHE/3DSignedWrithe_{knot_type}.dat.lp{pers_len}.dat.nos"
    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    fname_sta = f"SIGWRITHE/3DSignedWrithe_{knot_type}.dat.lp{pers_len}.dat.nos"
    STA = np.loadtxt(os.path.join(my_knot_dir, fname_sta))
    STA = STA.reshape(-1, Nbeads)
    return STA

@njit
def vassiliev_combinatorical_STS(STS):
    '''
    Calculate the Vassiliev invariants for a given knot
    '''
    samples = 5
    vassiliev1_data = []
    vassiliev2_data = []
    vassiliev3_data = []
    vassiliev4_data = []
    c = 6
    for idy in range(0, samples): # samples
        integral1 = 0
        integral2 = 0
        integral3 = 0
        integral4 = 0
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
                                            integral1 += STS[idy][i, m]*STS[idy][j, l]*STS[idy][k, n] # these are symmetry groups
                                            integral2 += STS[idy][i, j]*STS[idy][k, l]*STS[idy][m, n] 
                                            integral3 += STS[idy][i, k]*STS[idy][j, m]*STS[idy][l, n]
                                            integral4 += STS[idy][i, l]*STS[idy][j, m]*STS[idy][k, n]

        # self_linking = integral / (100 * 100 * 8 * math.pi)
        # vassiliev = (6 * self_linking) + (1/4)
        vassiliev1 = integral1 / (100 * 100 * 100)
        vassiliev1_data.append(vassiliev1)
        vassiliev2 = integral2 / (100 * 100 * 100)
        vassiliev3 = integral3 / (100 * 100 * 100)
        vassiliev4 = integral4 / (100 * 100 * 100)
        vassiliev2_data.append(vassiliev2)
        vassiliev3_data.append(vassiliev3)
        vassiliev4_data.append(vassiliev4)

    avg_vassiliev1 = sum(vassiliev1_data) / len(vassiliev1_data)
    avg_vassiliev2 = sum(vassiliev2_data) / len(vassiliev2_data)
    avg_vassiliev3 = sum(vassiliev3_data) / len(vassiliev3_data)
    avg_vassiliev4 = sum(vassiliev4_data) / len(vassiliev4_data)

    return vassiliev1_data, vassiliev2_data, vassiliev3_data, vassiliev4_data, avg_vassiliev1, avg_vassiliev2, avg_vassiliev3, avg_vassiliev4

@njit
def vassiliev_combinatorical_STA(STA, test_points, combinatorics, t):
    '''
    Calculate the Vassiliev invariants for a given knot
    '''
    samples = 1000
    vassiliev_data = []
    for idy in range(0, samples): # samples
        integral = 0
        for idx, i in enumerate(test_points):

            integral += STA[idy][i[0]]*STA[idy][i[1]]                                                

        self_linking = integral / (100 * 8 * math.pi)
        vassiliev = (6 * self_linking) + (1/4)
        vassiliev_data.append(vassiliev)

    avg_vassiliev = sum(vassiliev_data) / len(vassiliev_data)

    return avg_vassiliev, vassiliev_data

def main():
    knots = ["3_1"]
    avgs1 = []
    avgs2 = []
    avgs3 = []
    avgs4 = []
    for x in knots:
        STS = load_STS(x, 100, 10) # this is quite slow
        print("StS loaded")
        print("Calculating Vassiliev invariants...")
        v_d1, v_d2, v_d3, v_d4, av_1, av_2, av_3, av_4 = vassiliev_combinatorical_STS(STS)
        with open(f'vassiliev_{x}_comb_{6}_15_24_36.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in v_d1:
                writer.writerow([item])
        avgs1.append(av_1)

        with open(f'vassiliev_{x}_comb_{6}_12_34_56.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in v_d2:
                writer.writerow([item])
        avgs2.append(av_2)

        with open(f'vassiliev_{x}_comb_{6}_13_25_46.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in v_d3:
                writer.writerow([item])
        avgs3.append(av_3)

        with open(f'vassiliev_{x}_comb_{6}_14_25_36.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in v_d4:
                writer.writerow([item])
        avgs4.append(av_4)

    print(f"Combinatorics [152436]: ", avgs1)
    print(f"Combinatorics [123456]: ", avgs2)
    print(f"Combinatorics [132546]: ", avgs3)
    print(f"Combinatorics [142536]: ", avgs4)

    # for x in knots:
    #     print(x)
    #     STA = load_STA(x, 100, 10)
    #     print("StA loaded")
    #     test_points = combinations(indicies, 3)
    #     print("Combinations generated")
    #     print("Calculating Vassiliev invariants...")
    #     avg, v_d = vassiliev_combinatorical_STA(STA, test_points, 2, 1)
    #     with open(f'vassiliev_{x}_sta.csv', 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         for item in v_d:
    #             writer.writerow([item])
    #     avgs.append(avg)

    # print(f"Combinatorics: ", avgs)



main()






