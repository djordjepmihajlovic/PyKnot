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
    samples = 100
    vassiliev_data = []

    for idy in range(0, samples): # samples
        integral1 = 0
        integral2 = 0
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
                                            integral1 += STS[idy][i, k]*STS[idy][j, m]*STS[idy][l, n] 
                                            integral2 += STS[idy][i, l]*STS[idy][j, m]*STS[idy][k, n]

        # self_linking = integral / (100 * 100 * 8 * math.pi)
        # vassiliev = (6 * self_linking) + (1/4)
        integral = (0.5 * integral1) + integral2
        vassiliev = integral / (100 * 100 * 100 * -2)
        vassiliev_data.append(vassiliev)

    avg_vassiliev = sum(vassiliev_data) / len(vassiliev_data)

    return vassiliev_data, avg_vassiliev

@njit
def vassiliev_combinatorical_STA(STA):
    '''
    Calculate the Vassiliev invariants for a given knot
    '''
    samples = 100
    vassiliev_data = []
    for idy in range(0, samples): # samples
        integral = 0
        N = 100
        for i in range(0, N):
            for j in range(0, N):
                if i<j:
                    integral += STA[idy][j]*STA[idy][i]                                                

        self_linking = integral / (100* 100 * 8 * math.pi)
        vassiliev = (6 * self_linking) + (1/4)
        vassiliev_data.append(vassiliev)
        print(vassiliev)

    avg_vassiliev = sum(vassiliev_data) / len(vassiliev_data)

    return avg_vassiliev, vassiliev_data

def main():
    knots = ["3_1"]
    avgs = []
    # for x in knots:
    #     STS = load_STS(x, 100, 10) # this is quite slow
    #     print("StS loaded")
    #     print("Calculating Vassiliev invariants...")
    #     v_d, av = vassiliev_combinatorical_STS(STS)

    #     with open(f'vassiliev_{x}_v3.csv', 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         for item in v_d:
    #             writer.writerow([item])
    #     avgs.append(av)


    # print(f"Vassiliev: ", avgs)

    for x in knots:
        print(x)
        STA = load_STA(x, 100, 10)
        print("StA loaded")
        print("Combinations generated")
        print("Calculating Vassiliev invariants...")
        avg, v_d = vassiliev_combinatorical_STA(STA)
        # with open(f'vassiliev_{x}_sta.csv', 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     for item in v_d:
        #         writer.writerow([item])
        avgs.append(avg)

    print(f"Combinatorics: ", avgs)



main()

v2 = [0, 1, -1, 3, 2, -2, -1, 1, 6, 3, 5, 4, 4, 1, -1, -3, 0, -4, -3, -1, -2, 2, 2, -2, 3, -1, -3, 1, 0, 4, 1, -1, 1, 5, 2, 0]
v3 = [0, -1, 0, -5, -3, 1, 1, 0, -14, -6, 11, 8, -8, -2, -1, 3, 1, 0, 1, -3, 3, 2, 1, 0, 3, 2, 0, 1, 0, -7, -1, 0, 0, 10, -2, 1]
knots = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7", "8_1", "8_2", "8_3", "8_4", "8_5", "8_6", "8_7", "8_8", "8_9", "8_10", "8_11", "8_12", "8_13", "8_14", "8_15", "8_16", "8_17", "8_18", "8_19", "8_20", "8_21"]

list1 = []
for i in range(0, len(v2)):
    for j in range(0, len(v2)):
        if i != j:
            if v2[i] == v2[j]:
                print(f"v2: {i} {j}")
                list1.append([i, j])
                if [j, i] in list1:
                    list1.remove([j, i])

for i in list1:
    if v3[i[0]] == v3[i[1]]:
        print(knots[i[0]], knots[i[1]])








