from sklearn.externals import joblib
from processingdata import loadData
from data import Hand


def prepareData():
    images = []
    labels = []
    if (images is not None) and (labels is not None):
        images, labels = loadData()

    hand = Hand(images, labels)

    return hand

