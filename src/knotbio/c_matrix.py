import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit

@njit
def gen_c_matrix():
    '''
    Generate a 100x100 matrix of the form required for vassiliev invariants
    '''
    N = 100
    c_matrix_1 = numpy.zeros((N, N))
    c_matrix_2 = numpy.zeros((N, N))
    c_matrix_3 = numpy.zeros((N, N))


    for i in range(0, N):
        for j in range(0, N):
            if i<j:
                for k in range(0, N):
                    for l in range(0, N):
                        if j<k and k<l:
                            for m in range(0, N):
                                for n in range(0, N):
                                    if l<m and m<n:
                                        c_matrix_1[i][j] += 1   
                                        c_matrix_2[k][l] += 1
                                        c_matrix_3[m][n] += 1

    return c_matrix_1, c_matrix_2, c_matrix_3


x_1, x_2, x_3 = gen_c_matrix()
sns.heatmap(x_1, annot=False, cmap='viridis')
plt.title(r'$\omega_{ij}$ for $i<j<k<l<m<n$')
plt.savefig('comb_matrix_1(100C4).png')
plt.clf()

sns.heatmap(x_2, annot=False, cmap='viridis')
plt.title(r'$\omega_{kl}$ for $i<j<k<l<m<n$')
plt.savefig('comb_matrix_2(100C4).png')
plt.clf()

sns.heatmap(x_3, annot=False, cmap='viridis')
plt.title(r'$\omega_{mn}$ for $i<j<k<l<m<n$')
plt.savefig('comb_matrix_3(100C4).png')
plt.clf()



