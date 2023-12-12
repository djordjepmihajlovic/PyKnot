import os
import numpy as np
import matplotlib.pyplot as plt

# code for finding the dowker code of a 3d knot projected onto a 2d space


dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/"
fname = "XYZ/XYZ_3_1.dat.nos"

f = open(os.path.join(dirname, fname), "r")
len_db = 100000
knot_count = int(len_db/100)

count = 0
broken_knots = []
knots_multi_seg = []
n = 15
for knot_choice in range(n, n+1):
    knot_x = np.zeros((knot_count, 100))
    knot_y = np.zeros((knot_count, 100))
    knot_z = np.zeros((knot_count, 100))

    for j in range(0, int(len_db/100)-1):
        for i in range(0, 100):
            data = f.readline()
            point = [data]
            point = [float(value) for item in point for value in item.strip().split()]
            knot_x[j][i] = point[0]
            knot_y[j][i] = point[1]
            knot_z[j][i] = point[2]

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

        for j in range(0, 100): # looping backwards fixes some cases, breaks others!
            if j!= i and j!=(i+1)%100 and j!=(i-1)%100:
                vec_x3 = knot_x[knot_choice][j%100] 
                vec_x4 = knot_x[knot_choice][(j+1)%100] 
                vec_y3 = knot_y[knot_choice][j%100] 
                vec_y4 = knot_y[knot_choice][(j+1)%100] 

                # linear alg.
                t = ((vec_x1-vec_x3)*(vec_y3-vec_y4) - (vec_y1-vec_y3)*(vec_x3-vec_x4))/((vec_x1-vec_x2)*(vec_y3-vec_y4) - (vec_y1-vec_y2)*(vec_x3-vec_x4))
                s = ((vec_x1-vec_x3)*(vec_y1-vec_y2) - (vec_y1-vec_y3)*(vec_x1-vec_x2))/((vec_x1-vec_x2)*(vec_y3-vec_y4) - (vec_y1-vec_y2)*(vec_x3-vec_x4))


                if 0<s<1:
                    if 0<t<= 1:
                        crossings.append([i,j])
                        crossing_per_segment +=1
        x_seg.append(crossing_per_segment)

    for indx, i in enumerate(crossings):
        top = i[0]
        bottom = i[1]
        for indj, j in enumerate(crossings):
            if top == j[1] and bottom == j[0]:
                if indx%2 ==0:
                    gauss.append([indx+1, indj+1])
                    dowker.append(indj+1)

    for i in dowker:
        if i%2 == 1:
            broken_knots.append(knot_choice)

    for i in x_seg:
        if i>1:
            knots_multi_seg.append(knot_choice)

broken_knots = set(broken_knots)
knots_multi_seg = set(knots_multi_seg)


# print(crossings)
# print(x_seg)
print(f"gauss code: {gauss}")
print(f"dowker code: {dowker}")

knot_x_overcrossing = []
knot_y_overcrossing = []
knot_x_undercrossing = []
knot_y_undercrossing = []

for i in crossings:
    knot_x_overcrossing.append(knot_x[knot_choice][i[0]])
    knot_y_overcrossing.append(knot_y[knot_choice][i[0]])
    knot_x_undercrossing.append(knot_x[knot_choice][i[1]])
    knot_y_undercrossing.append(knot_y[knot_choice][i[1]])



fig = plt.figure()
ax = plt.axes(projection='3d')

## plot in 3D
# ax.plot3D(knot_x[knot_choice], knot_y[knot_choice], knot_z[knot_choice], 'r-')
# ax.plot3D([knot_x[knot_choice][0], knot_x[knot_choice][-1]], 
#         [knot_y[knot_choice][0], knot_y[knot_choice][-1]], 
#         [knot_z[knot_choice][0], knot_z[knot_choice][-1]], 'r-')

## projection in 2D
ax.plot(knot_x[knot_choice], knot_y[knot_choice], 'b-', zdir='z', zs=1.5)
ax.plot(knot_x_overcrossing, knot_y_overcrossing, 'gx', zdir = 'z', zs = 1.5)
ax.plot([knot_x[knot_choice][0], knot_x[knot_choice][-1]], [knot_y[knot_choice][0], knot_y[knot_choice][-1]], 'b-', zdir='z', zs=1.5)
for x, y, value in zip(knot_x_overcrossing, knot_y_overcrossing, dowker):
    ax.text(x, y, 1.5, str(value), color='black')

plt.show()
