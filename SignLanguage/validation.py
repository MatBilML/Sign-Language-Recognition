from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import sem
import numpy as np


def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(n_splits=K, random_state=33)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(np.mean(scores))
    print(sem(scores))
    #print("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


def accuracyScore(y, predict, model):
    print("accuracty of model %s:%d" % (model, accuracy_score(y, predict)))


def confusionMatrix(y, predict, model):
    print("confusion matrix of model %s:%d" % (model, confusion_matrix(y, predict)))
