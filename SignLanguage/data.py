
from helper import labelEncoder
import numpy as np

class Hand(object):

    def __init__(self, image, targetName):
        self.image = np.array(image)
        self.targetName = np.array(targetName)
        self.targetName=np.unique(self.targetName)

        self.data = self.reshape(self.image)
        self.target = self.encodeData(targetName)


    def reshape(self, data):
        length = len(data)
        return data.reshape(length, -1)

    def encodeData(self, targetName):
        return labelEncoder(targetName)
