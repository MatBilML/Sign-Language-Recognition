import utils
import processingdata
import cv2 as cv


def getImage(fileName):

    image = utils.read(fileName)

    feature = processingdata.extractColorHistogram(image)

    # image = utils.BGR2GRAY(image)

    cropImg = processingdata.preprocessing(image)

    cropImg = utils.resize(cropImg, 100, 100)
    cropImg = cropImg.reshape(1,-1)
    return cropImg, feature

"""
def getVideo():
    camera = cv.VideoCapture(0)

    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            cv.imshow("frame", frame)
"""