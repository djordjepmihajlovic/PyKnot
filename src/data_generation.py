import itertools
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from helper import *
import math
from matplotlib.ticker import AutoMinorLocator

class StA():
    def __init__(self, prob, test_data, train_data):

        self.prob = get_knots(prob)
        self.X_train = []
        self.y_train = []
        self.train_data = train_data
        for X, y in self.train_data:
            X = [i.numpy().flatten() for i in X]
            self.X_train.append(X)
            self.y_train.append(y.numpy())

        self.X_train = list(itertools.chain.from_iterable(self.X_train))
        self.y_train = list(itertools.chain.from_iterable(self.y_train))

        self.X_test = []
        self.y_test = []
        self.test_data = test_data
        for X, y in self.test_data:
            X = [i.numpy().flatten() for i in X]
            self.X_test.append(X)
            self.y_test.append(y.numpy())

        self.X_test = list(itertools.chain.from_iterable(self.X_test))
        self.y_test = list(itertools.chain.from_iterable(self.y_test))

    def calc_area(self):

        indicies = np.arange(0, 100, 1)

        area_0_1 = []
        area_3_1 = []
        area_4_1 = []
        area_5_1 = []
        area_5_2 = []
        area_6_1 = []
        area_6_2 = []
        area_6_3 = []
        area_7_1 = []
        area_7_2 = []
        area_7_3 = []

        for idx, X in enumerate(self.X_train):
            if self.y_train[idx] == 0:
                maxima, _ = find_peaks(X, prominence=1, width=5)
                area_0_1.append(np.trapz(y=X, x=indicies)/(6*math.pi))

            if self.y_train[idx] == 1: 
                area_3_1.append(np.trapz(y=X, x=indicies)/(6*math.pi))

            elif self.y_train[idx] == 2: 
                area_4_1.append(np.trapz(y=X, x=indicies)/(6*math.pi))

            elif self.y_train[idx] == 3:
                V = X
                area_5_1.append(np.trapz(y=X, x=indicies)/(6*math.pi))

            elif self.y_train[idx] == 4:
                area_5_2.append(np.trapz(y=X, x=indicies)/(6*math.pi))

            elif self.y_train[idx] == 5:
                L = X
                area_6_1.append(np.trapz(y=X, x=indicies))

            elif self.y_train[idx] == 6:
                L = X
                area_6_2.append(np.trapz(y=X, x=indicies))

            elif self.y_train[idx] == 7:
                L = X
                area_6_3.append(np.trapz(y=X, x=indicies))

            elif self.y_train[idx] == 8:
                L = X
                area_7_1.append(np.trapz(y=X, x=indicies))

            elif self.y_train[idx] == 9:
                L = X
                area_7_2.append(np.trapz(y=X, x=indicies)/(6*math.pi))

            elif self.y_train[idx] == 10:
                L = X
                area_7_3.append(np.trapz(y=X, x=indicies))

        area_0_1 = np.array(area_0_1)
        print(f"area mean of 0_1: {np.average(area_0_1)} with std: {np.std(area_0_1)}")

        area_3_1 = np.array(area_3_1)
        print(f"area mean of 3_1: {np.average(area_3_1)} with std: {np.std(area_3_1)}")

        area_4_1 = np.array(area_4_1)
        print(f"area mean of 4_1: {np.average(area_4_1)} with std: {np.std(area_4_1)}")

        area_5_1 = np.array(area_5_1)
        print(f"area mean of 5_1: {np.average(area_5_1)} with std: {np.std(area_5_1)}")

        area_5_2 = np.array(area_5_2)
        print(f"area mean of 5_2: {np.average(area_5_2)} with std: {np.std(area_5_2)}")

        area_7_2 = np.array(area_7_2)
        print(f"area mean of 7_2: {np.average(area_7_2)} with std: {np.std(area_7_2)}")

        sns.set_theme(style="white")

        # sns.histplot(area_0_1, color="purple", label= r'$0_{1}$', linewidth=0.1, edgecolor = "black")
        # sns.histplot(area_3_1, color="orange", label=r'$3_{1}$', linewidth=0.1, edgecolor = "black")
        # sns.histplot(area_4_1, color="blue", label=r'$4_{1}$', linewidth=0.1, edgecolor = "black")
        sns.histplot(area_5_1, color="red", label=r'$5_{1}$', linewidth=0.1, edgecolor = "black")
        # sns.histplot(area_5_2, color="green", label=r'$5_{2}$', linewidth=0.1, edgecolor = "black")
        sns.histplot(area_7_2, color="blue", label=r'$7_{2}$', linewidth=0.1, edgecolor = "black")

        plt.xlabel(r'Global Writhe, $Wr^{(i)}$')
        plt.ylabel("Frequency")
        plt.gca().tick_params(which="both", direction="in", right=True, top=True)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.legend()
        plt.show()

