import cv2 as cv
import numpy as np
import os
import utils


def preprocessing(image):
    value = (35, 35)
    #show(image)
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_bound = np.array([0, 40, 30], dtype="uint8")
    upper_bound = np.array([43, 255, 254], dtype="uint8")
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    res = cv.bitwise_and(image, image, mask=mask)
    res = cv.cvtColor(res, cv.COLOR_HSV2RGB)
    """

    res = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(res, value, 0)
    _, thresh1 = cv.threshold(blurred, 127, 255,
                              cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    #show(thresh1)
    conimage, contours, hierarchy = cv.findContours(thresh1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    maxArea = np.zeros(len(contours))
    if len(contours) != 0:
        for c in range(len(contours)):
            maxArea[c] = cv.contourArea(contours[c])

    index = maxArea.argmax()
    hand = contours[index]
    x, y, w, h = cv.boundingRect(hand)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    img = image[y:y + h, x:x + w]
    #show(img)
    return img

def extractColorHistogram(image, bins=(8, 8, 8)):
    hsv = utils.BGR2HSV(image)
    hist = cv.calcHist([hsv], [0, 1, 2], None, bins,
                       [0, 180, 0, 256, 0, 256])

    cv.normalize(hist, hist)

    return hist.flatten()


def make_skin(image):
    hsv = utils.BGR2HSV(image)

    lower_bound = np.array([0, 40, 30], dtype="uint8")
    upper_bound = np.array([43, 255, 254], dtype="uint8")
    mask = cv.inRange(hsv, lower_bound, upper_bound)
    res = cv.bitwise_and(image, image, mask=mask)
    res = utils.HSV2RGB(res)
    res = utils.BGR2GRAY(res)

    return res


def loadData(path="images"):
    images = []
    labels = []
    features = []
    filePath = [x[0] for x in os.walk(path)]

    for fileName in filePath[1:]:
        Name = os.listdir(fileName)
        for file in Name:

            label = fileName.split(os.path.sep)[-1].split("-")[0]
            image = utils.read(os.path.join(fileName, file))
            try:
                if image.dtype == np.uint8:
                    image=utils.BGR2GRAY(image)

                    images.append(image)
                    labels.append(label)
            except AttributeError as error:
                print(error)
            except IOError as error:
                print(error)

    return images, labels
