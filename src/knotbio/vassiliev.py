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
    samples = 100000 # number of knots to be calculated.
    vassiliev_data = []

    for idy in range(0, samples): # samples
        integral = 0
        integral1 = 0
        integral2 = 0
        N = 100
        for i in range(0, N):
            for j in range(0, N):
                if i>j:
                    for k in range(0, N):
                        for l in range(0, N):
                            if j>k and k>l:
                                # integral += STS[idy][i, k]*STS[idy][j, l]
                                for m in range(0, N):
                                    for n in range(0, N):
                                        if l<m and m<n:
                                            integral1 += STS[idy][i, k]*STS[idy][j, m]*STS[idy][l, n] 
                                            integral2 += STS[idy][i, l]*STS[idy][j, m]*STS[idy][k, n]

        # self_linking = integral / (100 * 100 * 8 * math.pi)
        # vassiliev = (6 * self_linking) + (1/4)
        integral = (0.5 * integral1) + integral2
        vassiliev = integral / (100 * 100 * 100 * -2) * 10/(4 * math.pi)
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
    knots = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7", "8_1", "8_2", "8_3", "8_4", "8_5", "8_6", "8_7", "8_8", "8_9", "8_10", "8_11", "8_12", "8_13", "8_14", "8_15", "8_16", "8_17", "8_18", "8_19", "8_20", "8_21", "9_1", "9_2", "9_3", "9_4", "9_5", "9_6", "9_7", "9_8", "9_9", "9_10", "9_11", "9_12", "9_13", "9_14", "9_15", "9_16", "9_17", "9_18", "9_19", "9_20", "9_21", "9_22", "9_23", "9_24", "9_25", "9_26", "9_27", "9_28", "9_29", "9_30", "9_31", "9_32", "9_33", "9_34", "9_35", "9_36", "9_37", "9_38", "9_39", "9_40", "9_41", "9_42", "9_43", "9_44", "9_45", "9_46", "9_47", "9_48", "9_49", "10_1", "10_2", "10_3", "10_4", "10_5", "10_6", "10_7", "10_8", "10_9", "10_10", "10_11", "10_12", "10_13", "10_14", "10_15", "10_16", "10_17", "10_18", "10_19", "10_20", "10_21", "10_22", "10_23", "10_24", "10_25", "10_26", "10_27", "10_28", "10_29", "10_30", "10_31", "10_32", "10_33", "10_34", "10_35", "10_36", "10_37", "10_38", "10_39", "10_40", "10_41", "10_42", "10_43", "10_44", "10_45", "10_46", "10_47", "10_48", "10_49", "10_50", "10_51", "10_52", "10_53", "10_54", "10_55", "10_56", "10_57", "10_58", "10_59", "10_60", "10_61", "10_62", "10_63", "10_64","10_65", "10_66", "10_67", "10_68", "10_69", "10_70", "10_71", "10_72", "10_73", "10_74", "10_75", "10_76", "10_77", "10_78", "10_79", "10_80", "10_81", "10_82", "10_83", "10_84", "10_85", "10_86", "10_87", "10_88", "10_89", "10_90", "10_91", "10_92", "10_93", "10_94", "10_95", "10_96", "10_97", "10_98", "10_99", "10_100", "10_101", "10_102", "10_103", "10_104", "10_105", "10_106", "10_107", "10_108", "10_109", "10_110", "10_111", "10_112", "10_113", "10_114", "10_115", "10_116", "10_117", "10_118", "10_119", "10_120", "10_121", "10_122", "10_123", "10_124", "10_125", "10_126", "10_127", "10_128", "10_129", "10_130", "10_131", "10_132", "10_133", "10_134", "10_135", "10_136", "10_137", "10_138", "10_139", "10_140", "10_141", "10_142", "10_143", "10_144", "10_145", "10_146", "10_147", "10_148", "10_149", "10_150", "10_151", "10_152", "10_153", "10_154", "10_155", "10_156", "10_157", "10_158", "10_159", "10_160", "10_161", "10_162", "10_163", "10_164", "10_165"]
    #knots = ["0_1"]
    avgs = []
    for x in knots:
        STS = load_STS(x, 100, 10) # this is quite slow
        print("StS loaded")
        print("Calculating Vassiliev invariants...")
        v_d, av = vassiliev_combinatorical_STS(STS)

        with open(f'vassiliev_{x}_v3_100000.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in v_d:
                writer.writerow([item])
        avgs.append(av)


    print(f"Vassiliev: ", avgs)

    # for x in knots:
    #     print(x)
    #     STA = load_STA(x, 100, 10)
    #     print("StA loaded")
    #     print("Combinations generated")
    #     print("Calculating Vassiliev invariants...")
    #     avg, v_d = vassiliev_combinatorical_STA(STA)
    #     # with open(f'vassiliev_{x}_sta.csv', 'w', newline='') as f:
    #     #     writer = csv.writer(f)
    #     #     for item in v_d:
    #     #         writer.writerow([item])
    #     avgs.append(avg)

    # print(f"Combinatorics: ", avgs)



main()


def test():

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








