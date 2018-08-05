import cv2
import numpy as np
import sys, os


def preprocessing(img, size=(48, 48)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    return img
