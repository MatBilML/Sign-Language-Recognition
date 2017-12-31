import traceback

import cv2 as cv
import numpy as np
import utils
from processingdata import preprocessing

#still working on camera prediction

def camera():
    camera = cv.VideoCapture(0)

    while camera.isOpened():
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture image")
            continue

        #frame = utils.resize(frame, 100, 100)
        cv.imshow("Camera", frame)

        print("hello")
        cropImg = preprocessing(frame)
        cv.imshow("cropImg", cropImg)

        cv.waitKey(200)
        #cv.destroyAllWindows()
    cv.destroyAllWindows()


def background():
    camera=cv.VideoCapture(0)
    back = cv.createBackgroundSubtractorMOG2()

    while camera.isOpened():
        ret,frame=camera.read()
        mask=back.apply(frame)
        if not ret:
            print("Hello")

        cv.imshow("Original",frame)
        cv.imshow("Mask",mask)
        cv.waitKey(20)
cv.destroyAllWindows()




if __name__ == "__main__":
    camera()
    #background()