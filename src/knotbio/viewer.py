import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoMinorLocator

knot_type = "3_1"

dirname = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data/"
fname_xyz = f"XYZ/XYZ_{knot_type}.dat.nos"
fname_sta = f"SIGWRITHE/3DSignedWrithe_{knot_type}.dat.lp{10}.dat.nos"


STA = open(os.path.join(dirname, fname_sta), "r")
XYZ = open(os.path.join(dirname, fname_xyz), "r")
X_dat = []
Y_dat = []
Z_dat = []
STA_dat = []

len_db = 100000
knot_count = int(len_db/100)
print(knot_count)
knot_num = 2
limit = 100

XYZ = open(f"../../knot data/ideal/3_1.csv", "r")

for i in range(0, limit):

    data = XYZ.readline()
    if i>limit-100:
        xyz = [data]
        xyz = [float(value) for item in xyz for value in item.strip().split()]
        X_dat.append(xyz[0])
        Y_dat.append(xyz[1])
        Z_dat.append(xyz[2])

for i in range(0, limit):
    data = STA.readline()
    if i>limit-100:
        if i>=1:
            point = [data]
            point = [float(value) for item in point for value in item.strip().split()]
            STA_dat.append(point[2])

z = np.arange(0, 99, 1)


sns.set_style("whitegrid")
plt.plot(STA_dat)
# plt.plot(ideal_4_1(z, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9]), 'r--')
print(STA_dat[29])
# plt.scatter(29, STA_dat[29], marker='x', color='r')   
# plt.scatter(49, STA_dat[49], marker='x', color='r')
# plt.scatter(87, STA_dat[87], marker='x', color='y')
# plt.scatter(58, STA_dat[58], marker='x', color='g')
# plt.scatter(18, STA_dat[18], marker='x', color='g')
# plt.scatter(75, STA_dat[75], marker='x', color='b')
# plt.scatter(1, STA_dat[1], marker='x', color='b')
# plt.scatter(37, STA_dat[37], marker='x', color='g')


plt.gca().tick_params(which="both", direction="in", right=True, top=True)
plt.xlabel("Bead number")
plt.ylabel("StA Writhe")
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.gca().xaxis.set_ticks_position('both')
plt.gca().yaxis.set_ticks_position('both')

plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d') 

## plot in 3D 

floor_Z = min(Z_dat) * np.ones(len(Z_dat)) * 3
floor_Y = min(Y_dat) * np.ones(len(Y_dat)) * 3
ax.plot3D(X_dat, Y_dat, Z_dat, 'b-', label='3D 3_1')
ax.plot3D([X_dat[-1], X_dat[0]], [Y_dat[-1], Y_dat[0]], [Z_dat[-1], Z_dat[0]], 'b-')
ax.plot3D(X_dat, Y_dat, floor_Z, 'r-', label='2D projection')
ax.plot3D([X_dat[-1], X_dat[0]], [Y_dat[-1], Y_dat[0]], [floor_Z[-1], floor_Z[0]], 'r-')
ax.plot3D(X_dat, floor_Y, Z_dat, 'r-')
ax.plot3D([X_dat[-1], X_dat[0]], [floor_Y[-1], floor_Y[0]], [Z_dat[-1], Z_dat[0]], 'r-')
ax.legend()

## 4_1 ideals
# ax.plot3D(X_dat[9], Y_dat[9], Z_dat[9], 'yo')
# ax.plot3D(X_dat[48], Y_dat[48], Z_dat[48], 'ro')
# ax.plot3D(X_dat[63], Y_dat[63], Z_dat[63], 'ro')
# ax.plot3D(X_dat[25], Y_dat[25], Z_dat[25], 'go')
# ax.plot3D(X_dat[84], Y_dat[84], Z_dat[84], 'go')
# ax.plot3D(X_dat[29], Y_dat[29], Z_dat[29], 'ro')
# ax.plot3D(X_dat[49], Y_dat[49], Z_dat[49], 'ro')
# ax.plot3D(X_dat[87], Y_dat[87], Z_dat[87], 'yo')
# ax.plot3D(X_dat[58], Y_dat[58], Z_dat[58], 'go')
# ax.plot3D(X_dat[18], Y_dat[18], Z_dat[18], 'go')
# ax.plot3D(X_dat[75], Y_dat[75], Z_dat[75], 'bo')
# ax.plot3D(X_dat[1], Y_dat[1], Z_dat[1], 'bo')
# ax.plot3D(X_dat[37], Y_dat[37], Z_dat[37], 'go')

## 3_1 ideals
# ax.plot3D(X_dat[52], Y_dat[52], Z_dat[52], 'ro')
# ax.plot3D(X_dat[90], Y_dat[90], Z_dat[90], 'ro')
# ax.plot3D(X_dat[29], Y_dat[29], Z_dat[29], 'ro')
# ax.plot3D(X_dat[66], Y_dat[66], Z_dat[66], 'go')
# ax.plot3D(X_dat[38], Y_dat[38], Z_dat[38], 'go')
# ax.plot3D(X_dat[15], Y_dat[15], Z_dat[15], 'go')

## 5_2 ideals
# ax.plot3D(X_dat[24], Y_dat[24], Z_dat[24], 'ro')
# ax.plot3D(X_dat[46], Y_dat[46], Z_dat[46], 'yo')
# ax.plot3D(X_dat[75], Y_dat[75], Z_dat[75], 'ro')
# ax.plot3D(X_dat[94], Y_dat[94], Z_dat[94], 'yo')
# ax.plot3D(X_dat[81], Y_dat[81], Z_dat[81], 'go')
# ax.plot3D(X_dat[60], Y_dat[60], Z_dat[60], 'go')
# ax.plot3D(X_dat[33], Y_dat[33], Z_dat[33], 'go')
# ax.plot3D(X_dat[11], Y_dat[11], Z_dat[11], 'go')

## 6_1 ideals
# ax.plot3D(X_dat[80], Y_dat[80], Z_dat[80], 'ro')
# ax.plot3D(X_dat[90], Y_dat[90], Z_dat[90], 'ro')
# ax.plot3D(X_dat[43], Y_dat[43], Z_dat[43], 'ro')
# ax.plot3D(X_dat[81], Y_dat[81], Z_dat[81], 'go')
# ax.plot3D(X_dat[60], Y_dat[60], Z_dat[60], 'go')
# ax.plot3D(X_dat[11], Y_dat[11], Z_dat[11], 'go')
# ax.plot3D(X_dat[68], Y_dat[68], Z_dat[68], 'go')

plt.show()