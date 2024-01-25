import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# code for finding the dowker code of a 3d knot projected onto a 2d space

knot_type = "3_1"

dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/"
fname = f"XYZ/XYZ_{knot_type}.dat.nos"

# 5_2, broken knot on n+5

f = open(os.path.join(dirname, fname), "r")
len_db = 100000
knot_count = int(len_db/100)
print(knot_count)

broken_knots = []
knots_multi_seg = []

# swapped [56, 97], [56, 5], [56, 95]
# unswapped [56, 5], [56, 95], [56, 97]

n = 0
dowker_dat = []

for knot_choice in range(n, n+100):
    knot_x = np.zeros((knot_count, 100))
    knot_y = np.zeros((knot_count, 100))
    knot_z = np.zeros((knot_count, 100))

    # for j in range(0, int(len_db/100)):
    for i in range(0, 100):
        data = f.readline() 
        point = [data]
        point = [float(value) for item in point for value in item.strip().split()]
        knot_x[knot_choice][i] = point[0]
        knot_y[knot_choice][i] = point[1]
        knot_z[knot_choice][i] = point[2]

    intersections = []
    inter_distance = []
    crossings = []
    dowker = []
    gauss = []
    cross_no = 0

    x_seg = []
    for i in range(0, 100):
        crossing_per_segment = 0
        vec_x1 = knot_x[knot_choice][i] 
        vec_x2 = knot_x[knot_choice][(i+1)%100] 
        vec_y1 = knot_y[knot_choice][i]
        vec_y2 = knot_y[knot_choice][(i+1)%100] 

        # plan, draw lines between each segment point. 
        #1->2, 1->3, ..., 1->100, 2->3, 2->4, ..., 2->100, ..., 99->100