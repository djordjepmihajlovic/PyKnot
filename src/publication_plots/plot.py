import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

    mse_t = 0
    error_percentage = 0
    modulo = []
    for idx, i in enumerate(avgs):
        mse = (i - true_vals[idx])**2
        mse_t += mse


    print(f'MSE: {mse_t/len(avgs)}')

    df_vassiliev = pd.DataFrame({
    r'measured ($I(K)_{(1,3)(2,4)}$)': avgs,  
    'true': true_vals
    }, index=knots)
    custom_palette = sns.color_palette("Set1")  

    sns.set_style("darkgrid")
    sns.set_palette(custom_palette)
    df_vassiliev.plot(kind = 'bar', rot= 0, width = 0.7, figsize=(10,5), linewidth=1, edgecolor = "black")
    plt.xlabel("Knot Type")
    plt.ylabel(r'$2^{nd}$-Order Vassiliev Invariant ($v_{2}$)')
    # plt.title(r'$I(K)_{(1,3)(2,4)} = \oint\oint\oint\oint_{K}\frac{r_{1}-r_{3}}{|r_{1}-r_{3}|^{3}} \cdot \frac{r_{2}-r_{4}}{|r_{2}-r_{4}|^{3}} \approx \sum_{1<2<3<4}\omega_{StS}[1][3]*\omega_{StS}[2][4]$', y=1.05)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.show()

plot_vassiliev()
