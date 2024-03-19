import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

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

for knot_choice in range(n, n+1):
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

        for j in range(0, 100): # looping forward
            if j!= i and j!=(i+1)%100 and j!=(i-1)%100:
                vec_x3 = knot_x[knot_choice][j%100] 
                vec_x4 = knot_x[knot_choice][(j+1)%100] 
                vec_y3 = knot_y[knot_choice][j%100] 
                vec_y4 = knot_y[knot_choice][(j+1)%100] 

                # linear alg.
                t = ((vec_x1-vec_x3)*(vec_y3-vec_y4) - (vec_y1-vec_y3)*(vec_x3-vec_x4))/((vec_x1-vec_x2)*(vec_y3-vec_y4) - (vec_y1-vec_y2)*(vec_x3-vec_x4))
                s = ((vec_x1-vec_x3)*(vec_y1-vec_y2) - (vec_y1-vec_y3)*(vec_x1-vec_x2))/((vec_x1-vec_x2)*(vec_y3-vec_y4) - (vec_y1-vec_y2)*(vec_x3-vec_x4))

                if 0<=s<=1:
                    if 0<=t<= 1:
                        crossings.append([i,j])
                        crossing_per_segment +=1

                        inter_x = vec_x1 + t * (vec_x2-vec_x1)
                        inter_y = vec_y1 + t * (vec_y2-vec_y1)

                        x = abs(inter_x - vec_x1)
                        y = abs(inter_y - vec_y1)
                        intersections.append([inter_x, inter_y]) # normalized distance along segment
                        inter_distance.append([(x**2+y**2)**(1/2), i])

        x_seg.append(crossing_per_segment)

    # works now

    for indx, i in enumerate(crossings):
        for indy, j in enumerate(crossings):
            if i[0] == j[0] and indy > indx:
                if inter_distance[indx] > inter_distance[indy]:
                    crossings[indx], crossings[indy] = crossings[indy], crossings[indx]


    for indx, i in enumerate(crossings):
        for indj, j in enumerate(crossings):
            if indj != indx:
                if i[0] == j[1] and i[1] == j[0]: 
                    gauss.append([indx+1, indj+1])
                    if indx%2 ==0:
                        dowker.append(indj+1)

    for i in dowker:
        if i%2 == 1:
            broken_knots.append(knot_choice)

    print(f'knot: {knot_choice}')

    dowker_dat.append(dowker)

## code for writing dowker code to csv file
    
# with open(f'../../knot data/dowker_{knot_type}.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(dowker_dat) 
    
broken_knots = set(broken_knots)

print(f"there are: {len(broken_knots)} broken knots")
print(f"broken knots: {broken_knots}")

print(f"interpoints: {inter_distance}")
print(f"crossing locations: {crossings}")
print(f"gauss code: {gauss}")
print(f"dowker code: {dowker}")

knot_x_overcrossing = []
knot_y_overcrossing = []
knot_x_undercrossing = []
knot_y_undercrossing = []

for indx, i in enumerate(crossings):
    if indx%2 == 0:
        knot_x_overcrossing.append(knot_x[knot_choice][i[0]])
        knot_y_overcrossing.append(knot_y[knot_choice][i[0]])
        knot_x_undercrossing.append(knot_x[knot_choice][i[1]])
        knot_y_undercrossing.append(knot_y[knot_choice][i[1]])


sns.set_theme()
fig = plt.figure()
ax = plt.axes(projection='3d') 

## plot in 3D 
# ax.plot3D(knot_x[knot_choice], knot_y[knot_choice], knot_z[knot_choice], 'b-')
# ax.plot3D([knot_x[knot_choice][0], knot_x[knot_choice][-1]], 
#         [knot_y[knot_choice][0], knot_y[knot_choice][-1]], 
#         [knot_z[knot_choice][0], knot_z[knot_choice][-1]], 'b-')


## k-means clustering algorithm to try produce knot graph

# data_kmeans = np.column_stack((knot_x[knot_choice], knot_y[knot_choice]))
# kmeans = KMeans(n_clusters=8)
# kmeans.fit(data_kmeans)
# center_points = kmeans.cluster_centers_

# ax.scatter(center_points[:, 0], center_points[:, 1], c='red', marker='X', s=25, label='Center Points')


## projection in 2D
ax.plot(knot_x[knot_choice], knot_y[knot_choice], 'r-')
ax.plot(knot_x_overcrossing, knot_y_overcrossing, 'gx')
ax.plot([knot_x[knot_choice][0], knot_x[knot_choice][-1]], [knot_y[knot_choice][0], knot_y[knot_choice][-1]], 'r-')
for x, y, value in zip(knot_x_overcrossing, knot_y_overcrossing, dowker):
    ax.text(x, y, 0, str(value), color='black')

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Remove tick marks
ax.tick_params(axis='x', which='both', bottom=False, top=False)
ax.tick_params(axis='y', which='both', left=False, right=False)
ax.tick_params(axis='z', which='both', bottom=False, top=False)

plt.tight_layout()
plt.show()


