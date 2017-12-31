import numpy as np
import cv2 as cv


def read(fileName):
    if fileName is not None:
        return cv.imread(fileName)
    return


def resize(image, height, width):
    return cv.resize(image, (height, width))


def BGR2GRAY(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def GRAY2BGR(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

def BGR2HSV(image):
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)

def HSV2RGB(image):
    return cv.cvtColor(image,cv.COLOR_HSV2RGB)

def crop(img, x1, x2, y1, y2):
    crp = img[y1:y2, x1:x2]
    crp = resize(crp, ((128, 128))) # resize
    return crp
