import splitdata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import predictionOfImage
import numpy as np
from preparedata import prepareData
from validation import evaluate_cross_validation
import os


class KNeighbors(object):
    def __init__(self):
        self.hand = prepareData()
        self.model = KNeighborsClassifier()
        self.X_train, self.X_test, self.Y_train, self.Y_test = splitdata.train_test_split(self.hand.data,
                                                                                          self.hand.target)

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, image):
        return self.hand.targetName[self.model.predict(image)]

    def score(self):
        print("Accuracy of model %s" % self.model.score(self.X_test, self.Y_test))

    def crossValidation(self):
        evaluate_cross_validation(self.model, self.X_train, self.Y_train, 5)

"""
class Svm(object):
    def __init__(self):
        self.hand = prepareData()
        self.model = SVC()
        self.X_train, self.X_test, self.Y_train, self.Y_test = splitdata.train_test_split(self.hand.data,
                                                                                          self.hand.target)

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, image):
        return self.hand.targetName[self.model.predict(image)]

    def score(self):
        print("Accuracy of model %s" % self.model.score(self.X_test, self.Y_test))

    def crossValidation(self):
        evaluate_cross_validation(self.model, self.X_train, self.Y_train, 5)


class Logistic(object):
    def __init__(self):
        self.hand = prepareData()
        self.model = LogisticRegression()
        self.X_train, self.X_test, self.Y_train, self.Y_test = splitdata.train_test_split(self.hand.data,
                                                                                          self.hand.target)

    def fit(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, image):
        return self.hand.targetName[self.model.predict(image)]

    def score(self):
        print("Accuracy of model %s" % self.model.score(self.X_test, self.Y_test))

    def crossValidation(self):
        evaluate_cross_validation(self.model, self.X_train, self.Y_train, 5)

"""
import utils

if __name__ == "__main__":
    knn = KNeighbors()
    knn.fit()
    print("Original value\tPredicted value")
    for file in os.listdir("tests"):
        file=os.path.join("tests",file)

        label = file.split(os.path.sep)[-1].split(".")[0]
        img, feature = predictionOfImage.getImage(file)

        predict = knn.predict(img)
        print("%s\t%s" % (label, predict))

    """
    Testing purpose data function depend on use
    """
    # knn.score()
    # knn.crossValidation()
