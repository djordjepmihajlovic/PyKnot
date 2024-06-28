import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator


def load_STS(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")

    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"
    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    STS = np.loadtxt(os.path.join(master_knots_dir, fname_sts))
    # STS = np.loadtxt(os.path.join(my_knot_dir, fname_sts))
    STS = STS.reshape(-1, Nbeads, Nbeads)
    return STS

def load_STA(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")

    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    fname_sta = f"SIGWRITHE/3DSignedWrithe_{knot_type}.dat.lp{pers_len}.dat.nos"
    # STA = np.loadtxt(os.path.join(my_knot_dir, fname_sta))

    f = open(os.path.join(my_knot_dir, fname_sta), "r")

    len_db = 100000
    knot_count = int(len_db/100)
    knot_count = np.arange(0, knot_count, 1)
    j = 0

    sta = [[] for x in knot_count]
    for i in range(0, len_db):
        data = f.readline() 
        if i>=1:
            point = [data]
            point = [float(value) for item in point for value in item.strip().split()]
            sta[j].append(point[2])
            if i % 100 == 0:
                j += 1

    return sta

def load_XYZ(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    master_knots_dir = "/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/"
    master_knots_dir = os.path.join(master_knots_dir,knot_type,f"N{Nbeads}",f"lp{pers_len}")

    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    fname_xyz = f"XYZ/XYZ_{knot_type}.dat.nos"
    XYZ = np.loadtxt(os.path.join(my_knot_dir, fname_xyz))
    XYZ = XYZ.reshape(-1, Nbeads, 3)
    return XYZ

knot_type = "3_1"

Nbeads = 100
pers_len = 10
x = np.arange(0, Nbeads, 1)

STS = load_STS(knot_type, Nbeads, pers_len)
XYZ = False
STA = False


sns.set_theme(style="white")
plt.imshow(STS[0], cmap='viridis')
plt.colorbar()
plt.xlabel(r'Segment $(x)$')
plt.ylabel(r'Segment $(y)$')
plt.savefig('STS_3_1.png')
plt.clf()

if STA == True:
    sns.set_theme(style="white")
    plt.plot(x, STA[0], 'b')
    plt.xlabel(r'Segment $(x_{i})$')
    plt.ylabel(r'$\omega_{StA}$')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.title(r'$\omega_{StA}$')
    plt.savefig('STA.png')
    plt.clf()

if XYZ == True:
    ax = plt.axes(projection='3d') 
    ax.plot(XYZ[0][:,0], XYZ[0][:,1], XYZ[0][:,2])
    plt.title('3D Knot Configuration')
    plt.savefig('XYZ_5_2.png')
    plt.clf()








