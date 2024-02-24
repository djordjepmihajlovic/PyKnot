import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

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
