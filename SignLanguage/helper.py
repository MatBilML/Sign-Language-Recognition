
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from  sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


def labelEncoder(labels):
    model = LabelEncoder()
    if labels is not None:
        return model.fit_transform(labels)
    return


def minMaxScaler(images, labels):
    model = MinMaxScaler()
    if (images is not None) and (labels is not None):
        model.fit_transform(images, labels)
    return


def standardScaler(images, labels):
    model = StandardScaler()
    if (images is not None) and (labels is not None):
        model.fit_transform(images, labels)
    return


def robustScaler(images, labels):
    model = RobustScaler()
    if (images is not None) and (labels is not None):
        model.fit_transform(images, labels)
    return
