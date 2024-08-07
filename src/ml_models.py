import numpy as np
import itertools
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns
from helper import *

class DecisionTree:
    def __init__(self, prob, train_data, test_data):
        self.prob = prob
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

    def classification(self):


        DT = Path(f"../trained models/DT_{self.prob}.sav")

        if DT.is_file() == False:
            print("training...")
            clf = tree.DecisionTreeClassifier(max_depth=20)
            clf = clf.fit(self.X_train, self.y_train)
            filename = f'DT_{self.prob}.sav'
            pickle.dump(clf, open(filename, 'wb'))

        print("loading trained model...")

        clf = pickle.load(open(f'DT_{self.prob}.sav', 'rb'))

        #clf = pickle.load(open(f'../trained models/DT_{self.prob}.sav', 'rb'))

        y_pred = clf.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred)
        conf_mat = confusion_matrix(self.y_test, y_pred)
        print(f"Accuracy: {score*100}%")
        print(conf_mat)
        ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=get_knots(self.prob)).plot()
        plt.show()

        TREE = Path(f"../trained models/tree_{self.prob}.log")
        if TREE.is_file() == True:
            text_rep = tree.export_text(clf)
            with open(f"tree_{self.prob}.log", "w") as fout:
                fout.write(text_rep)

        test_point = self.X_train[2]
        prediction = clf.predict(test_point.reshape(1, -1))
        print(self.y_train[2])
        print(prediction)
        decision_path = clf.decision_path(test_point.reshape(1, -1)).toarray()[0]
        tree_structure = clf.tree_
        importance = (clf.feature_importances_)

        return tree_structure, importance, test_point, decision_path







        
        














