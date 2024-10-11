import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from matplotlib.ticker import AutoMinorLocator

def beta_convergence():
    beta_convergence_5Class = {1: [6997.970, 5674.826, 5046.596, 4663.250, 4476.257, 4473.884], 
                            2: [9509.934, 8261.114, 7857, 7238, 6864,  6772.418, 6651, 6762.012], 
                            3: [10161.991, 9357.779, 8607.780, 8395.746, 8295.021, 8261.953],
                            4: [10225.843, 9793.581, 9635.809, 9500.890, 9490.114],
                            5: [11050, 10852, 10750, 10564.326, 10501]}

    sns.set_style("whitegrid")
    points_1 = [2, 4, 6, 8, 10, 12]
    points_2 = [1, 2, 3, 4, 6, 7, 10, 12]
    points_3 = [1, 2, 4, 6, 8, 10]
    points_4 = [2, 4, 6, 8, 10]
    points_5 = [2, 3, 4, 6, 10]
    plt.plot(points_3, beta_convergence_5Class[3])
    plt.gca().tick_params(which="both", direction="in", right=True, top=True)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.ylabel('MSE reconstruction loss: $\\frac{1}{n}\sum^{n}_{i=1}(y_{i}-\hat{y_{i}})^{2}$')
    plt.xlabel('Latent Space Dimension $\hat{z_{i}}$')
    plt.show()

def plot_vassiliev():   
    knots = [r'$0_{1}$', r'$3_{1}$', r'$4_{1}$', r'$5_{1}$', r'$5_{2}$', r'$6_{1}$', r'$6_{2}$', r'$6_{3}$', r'$7_{1}$', r'$7_{2}$', r'$7_{3}$', r'$7_{4}$', r'$7_{5}$', r'$7_{6}$', r'$7_{7}$', r'$8_{1}$', r'$8_{2}$', r'$8_{3}$', r'$8_{4}$', r'$8_{5}$', r'$8_{6}$', r'$8_{7}$', r'$8_{8}$', r'$8_{9}$', r'$8_{10}$', r'$8_{11}$', r'$8_{12}$', r'$8_{13}$', r'$8_{14}$', r'$8_{15}$', r'$8_{16}$', r'$8_{17}$', r'$8_{18}$', r'$8_{19}$', r'$8_{20}$', r'$8_{21}$']
    avgs = [0.2600493769828114, 1.1474045297302526, -0.5075482921458216, 2.924233072632151, 2.0914421230692413, -1.3066271919064858, -0.5210489061899892, 0.954239657893129, 5.612545617759219, 3.0556949856514577, 4.7771516455272165, 3.9652837760158013, 3.8834814369077257, 1.2230451933895257, -0.289236916156298, -2.0598934766954096, 0.36796170990400917, -2.9701283042425413, -2.2312954493641706, -0.5667962826916177, -1.2861794864333098, 1.6925320257979684, 1.8468487861397274, -1.513546466501713, 2.585913250758804, -0.40826133994813835, -2.0606469059373635, 0.9763627489407, 0.44702916528575565, 3.909511980427651, 0.7077559020130083, -0.6256123942033112, 0.2708967776991869, 4.732607420983808, 1.8408075088138254, 0.3047809344807912]
    true_vals =  [0, 1, -1, 3, 2, -2, -1, 1, 6, 3, 5, 4, 4, 1, -1, -3, 0, -4, -3, -1, -2, 2, 2, -2, 3, -1, -3, 1, 0, 4, 1, -1, 1, 5, 2, 0]
    true_vals_v3 = [0, -1, 0, -5, -3, 1, 1, 0, -14, -6, 11, 8, -8, -2, -1, 3, 1, 0, 1, -3, 3, 2, 1, 0, 3, 2, 0, 1, 0, -7, -1, 0, 0, 10, -2, 1]
    knots_v3 = [r'$0_{1}$', r'$3_{1}$', r'$4_{1}$', r'$5_{1}$', r'$5_{2}$', r'$6_{1}$', r'$6_{2}$', r'$6_{3}$', r'$7_{1}$', r'$7_{2}$', r'$7_{3}$']
    avgs_v3 = [0.0002337558283305493, -1.06775154096688, -0.021786476525896464, -5.75605466684332, -3.353080405326023, 1.7640745159354427, 1.3740907435768093, 0.023793775923153206, -16.516180240850172, -6.908003459709753, 12.785595712520207, 9.672215816049722, -9.27381714597119, -1.7617876763989577, -2.197872722531685, 4.026412506546035, 2.0788303085312414, 0.002621685470809658, 1.8529121793350916, -3.5166194377336013, 3.9306835723933795, 1.4584257439455035, 0.9582734930250816, -0.6395832680539215, 3.5848503429889265, 4.330628247278777, -0.19987925684603294, 0.5856539225972239, 0.362647377761927, -7.173200771490417, -1.1552579068252207, 0.018787389819954516, -0.011594225256305896, 10.346397228487199, -2.7876109122529646, 1.7680922142601654]
    avgs_v3 = [i*(10/(4 * math.pi)) for i in avgs_v3]


    mse_t = 0
    error_percentage = 0
    modulo = []
    for idx, i in enumerate(avgs_v3):
            mse_t += (i-true_vals_v3[idx])**2
    
    mse_t/=len(avgs)
    print(mse_t)


    print(f'MSE: {mse_t/len(avgs)}')

    df_vassiliev = pd.DataFrame({
    r'measured ($I(K)_{(1,3)(2,4)}$)': avgs, 
    'true': true_vals
    }, index=knots)
    custom_palette = sns.color_palette("Set1")  

    indexes = [avgs_v3.index(x) for x in sorted(avgs_v3)]
    print(indexes)

    sns.set_style("white")
    sns.set_palette(custom_palette)
    # df_vassiliev.plot(kind = 'bar', rot= 0, width = 0.7, figsize=(10,5), linewidth=1, edgecolor = "black")
    plt.xlabel("Knots (unsorted)")
    plt.ylabel(r'$3^{rd}$-Order Vassiliev Invariant ($v_{3}$)')
    # # plt.title(r'$I(K)_{(1,3)(2,4)} = \oint\oint\oint\oint_{K}\frac{r_{1}-r_{3}}{|r_{1}-r_{3}|^{3}} \cdot \frac{r_{2}-r_{4}}{|r_{2}-r_{4}|^{3}} \approx \sum_{1<2<3<4}\omega_{StS}[1][3]*\omega_{StS}[2][4]$', y=1.05)
    plt.plot(sorted(true_vals_v3), label='True', linestyle='-', color='black')
    plt.scatter(range(len(avgs_v3)), sorted(avgs_v3), label='Measured', marker='x')
    for i in range(0, len(sorted(avgs_v3)), 10):
        plt.text(i-0.5, sorted(avgs_v3)[i]+0.2, f'{knots[indexes[i]]}', ha='right', va='bottom', fontsize=12)   
        circle = plt.Circle((i, sorted(avgs_v3)[i]), 0.5, color='black', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.show()

def v_variance():
    '''
    Load the measured v2 data for some knot type
    '''

    f = open("../../../sample_data/vassiliev/vassiliev_7_5_v3_10000.csv", "r")


    data = []

    for i in f:
        data.append(float(i)/(-2*math.pi))
    
    n_bins = 20

    data_array = np.array(data)
    print(f'Mean: {np.mean(data_array)}')
    print(f'Std: {np.std(data_array)}')

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(data, bins=n_bins)

    plt.gca().tick_params(which="both", direction="in", right=True, top=True)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')

    plt.ylabel('Frequency')
    plt.xlabel(r'$3^{rd}$-Order Vassiliev Invariant ($v_{3}$) measure')

    plt.title('PLACEHOLDER - v3 (single knot/6_2)')

    plt.show()

def total_v_variance():
    '''
    Compare all v2 and v3 measurements via box plots
    '''

    knots = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3"]
    data = [[], [], [], [], [], [], [], [], [], [], []]

    for idx, i in enumerate(knots):

        # f = open(f"../../knot data/vassiliev/vassiliev_{i}_comb_4.csv", "r")
        f = open(f"../../knot data/vassiliev/vassiliev_{i}_v3.csv", "r") # v3


        for i in f:
            data[idx].append((float(i)) * 10/(4 * math.pi))

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    # generate some random test data
    all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    # plot violin plot
    axs.violinplot(data,
                    showmeans=False,
                    showmedians=True)
    
    true_v2 = [0, 1, -1, 3, 2, -2, -1, 1, 6, 3, 5]
    true_v3 = [0, -1, 0, -5, -3, 1, 1, 0, -14, -6, 11]

    inds = np.arange(1, len(true_v3) + 1)
    axs.scatter(inds, true_v3, marker='x', color='red', s=30, zorder=3)

    # adding horizontal grid lines
    axs.yaxis.grid(True)
    axs.xaxis.grid(True)
    axs.set_xlabel('Knots')
    axs.set_ylabel(r'$v_{3}$ Measurement')

    plt.title('PLACEHOLDER - v3')

    plt.show()



v_variance()
