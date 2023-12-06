import itertools
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from helper import *

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
                area_0_1.append(np.trapz(y=X, x=indicies))

            if self.y_train[idx] == 1: 
                area_3_1.append(np.trapz(y=X, x=indicies))

            elif self.y_train[idx] == 2: 
                area_4_1.append(np.trapz(y=X, x=indicies))

            elif self.y_train[idx] == 3:
                V = X
                area_5_1.append(np.trapz(y=X, x=indicies))

            elif self.y_train[idx] == 4:
                area_5_2.append(np.trapz(y=X, x=indicies))

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
                area_7_2.append(np.trapz(y=X, x=indicies))

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

        area_6_1 = np.array(area_6_1)
        print(f"area mean of 6_1: {np.average(area_6_1)} with std: {np.std(area_6_1)}")

        area_6_2 = np.array(area_6_2)
        print(f"area mean of 6_2: {np.average(area_6_2)} with std: {np.std(area_6_2)}")

        area_6_3 = np.array(area_6_3)
        print(f"area mean of 6_3: {np.average(area_6_3)} with std: {np.std(area_6_3)}")

        area_7_1 = np.array(area_7_1)
        print(f"area mean of 7_1: {np.average(area_7_1)} with std: {np.std(area_7_1)}")

        area_7_2 = np.array(area_7_2)
        print(f"area mean of 7_2: {np.average(area_7_2)} with std: {np.std(area_7_2)}")

        area_7_3 = np.array(area_7_3)
        print(f"area mean of 7_3: {np.average(area_7_3)} with std: {np.std(area_7_3)}")

        # sns.histplot(area_0_1, color="purple", label= "0_1", linewidth=0.1, edgecolor = "black")
        # sns.histplot(area_3_1, color="orange", label="3_1", linewidth=0.1, edgecolor = "black")
        # sns.histplot(area_4_1, color="blue", label="4_1", linewidth=0.1, edgecolor = "black")
        # sns.histplot(area_5_1, color="red", label="5_1", linewidth=0.1, edgecolor = "black")
        # sns.histplot(area_5_2, color="green", label="5_2", linewidth=0.1, edgecolor = "black")
        sns.histplot(area_6_2, color="yellow", label="6_2", linewidth=0.1, edgecolor = "black")
        plt.xlabel("AUC")
        plt.legend()
        plt.show()



    def calc_maxima(self):

        indicies = np.arange(0, 100, 1)

        area_0_1 = []
        area_3_1 = []
        area_4_1 = []
        area_5_1 = []
        area_5_2 = []
        area_6_1 = []
        area_6_2 = []

        for idx, X in enumerate(self.X_train):
            if self.y_train[idx] == 1: # 3_1
                grad = []
                for index, element in enumerate(X):
                    y_diff = X[(index+1)%100]-element
                    x_diff = indicies[(index+1)%100]-indicies[index]

                    grad.append(y_diff/x_diff)

                plt.plot(indicies, grad)
                plt.show()

                break


class XYZ():
    def __init__(self, prob, test_data, train_data):
        
        self.prob = get_knots(prob)
        self.X_x_train = []
        self.X_y_train = []
        self.X_z_train = []
        self.y_train = []
        self.train_data = train_data
        for X, y in self.train_data:
            X = [i.numpy().flatten() for i in X]
            break
            