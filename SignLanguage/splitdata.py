
from sklearn.model_selection import train_test_split

def splitData(images,labels):

    if (images is not None) and (labels is not None):
        return train_test_split(images,labels,test_size=.20,random_state=33)

    return