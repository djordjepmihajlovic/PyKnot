import numpy as np
import itertools
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns

# going to set up decision tree, linear regression and logistic regression predictors. Here; check results and interpretability...
# idea, assign color to point if it is indicating positive prediction for class

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
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(self.X_train, self.y_train)
            filename = f'DT_{self.prob}.sav'
            pickle.dump(clf, open(filename, 'wb'))

        print("loading trained model...")

        clf = pickle.load(open(f'../trained models/DT_{self.prob}.sav', 'rb'))

        y_pred = clf.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {score*100}%")

        TREE = Path(f"../trained models/tree_{self.prob}.log")
        if TREE.is_file() == False:
            text_rep = tree.export_text(clf)
            with open(f"tree_{self.prob}.log", "w") as fout:
                fout.write(text_rep)

        test_point = self.X_train[0]
        decision_path = clf.decision_path(test_point.reshape(1, -1)).toarray()[0]
        tree_structure = clf.tree_
        importance = (clf.feature_importances_)

        x = np.arange(0, 100)  
        node_index = 0
        features = []
        thresholds = []

        for i in range(tree_structure.node_count):
            # Check if the node is part of the decision path
            if decision_path[i] == 1:
                # Get feature index and threshold for the decision at this node
                feature_index = tree_structure.feature[node_index]
                threshold = tree_structure.threshold[node_index]
                
                # print(f"Node {node_index}: Feature {feature_index} <= {threshold}")
                features.append(feature_index)
                thresholds.append(threshold)
                
                # Determine the direction of the decision (left or right)
                decision = "left" if test_point[feature_index] <= threshold else "right"
                
                # Move to the next node based on the decision
                if decision == "left":
                    node_index = tree_structure.children_left[node_index]
                else:
                    node_index = tree_structure.children_right[node_index]

        features = np.unique(np.abs(features))

        test_point_DT = [i for idx,i in enumerate(test_point) if idx in features]

        sns.set_theme()
        ax = sns.barplot(x=x, y=importance, color='blue')
        ax.set_xticklabels([])

        plt.xlabel("Bead index")
        plt.ylabel("Relative importance")
        plt.title("Decision Tree Feature Importance")
        plt.show()

        plt.scatter(x, test_point, marker='.', label = "Base")
        plt.scatter(features, test_point_DT, marker='x', label = "DT nodes")
        print(features)

        for i, (x, y) in enumerate(zip(features, test_point_DT)):
            plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(5,5), ha='center')

        plt.legend()
        plt.xlabel("Bead index")
        plt.ylabel("StA Writhe")
        plt.title("DT model +ve prediction")

        plt.show()




        
        














