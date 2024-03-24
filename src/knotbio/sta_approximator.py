import os
import numpy as np
from numba import njit
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

def load_STS(knot_type, Nbeads, pers_len):
    '''
    Load the data from the knots database
    '''

    fname_sts = f"SIGWRITHEMATRIX/3DSignedWritheMatrix_{knot_type}.dat.lp10.dat"
    my_knot_dir = "/Users/djordjemihajlovic/Desktop/Theoretical Physics/MPhys/Data"
    STS = np.loadtxt(os.path.join(my_knot_dir, fname_sts))
    STS = STS.reshape(-1, Nbeads, Nbeads)
    return STS

@njit
def STS_STA_approx(STS):
    '''
    Calculate sta eta equivalence for a given knot
    '''
    idy = 0
    integral1 = []
    integral2 = []
    sta = []

    N = 100
    for i in range(0, N):
        integral_sum1 = 0
        integral_sum2 = 0
        integral_sta = 0
        for j in range(0, N):
            if i<j:
                integral_sum1 += STS[idy][j, i]

            integral_sta += STS[idy][i, j]

        integral1.append(integral_sum1/(100))
        integral2.append(integral_sum2/(100))
        sta.append(integral_sta/100)

    return integral1, sta

def main():
    knots = ["3_1"]
    avgs = []
    dat = np.arange(0, 100, 1)
    for x in knots:
        STS = load_STS(x, 100, 10) # this is quite slow
        print("StS loaded")
        print("Calculating Vassiliev invariants...")
        integral1, sta = STS_STA_approx(STS)
    sns.set_theme(style="white")
    plt.plot(integral1, label=r'$\epsilon_{StA}$')  
    plt.plot(sta, label=r'$\omega_{StA}$')
    plt.xlabel(r'Bead index, $x_{i}$')
    plt.legend()
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.show()


main()
