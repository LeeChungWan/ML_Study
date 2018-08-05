import cv2
import numpy as np
import sys, os


def preprocessing(img, size=(48, 48)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size).astype(np.float32)
    return img


def extract_features(path):
    X, y = [], []
    label = 0
    for dirnames in os.listdir(path):
        sub_path = os.path.join(path, dirnames)
        for filename in os.listdir(sub_path):
            file_path = os.path.join(sub_path, filename)
            img = cv2.imread(file_path)
            img = preprocessing(img)
            X.append(img)

            class_label = [0, 0]
            class_label[label] = 1
            y.append(class_label)
        label += 1
    X = np.asanyarray(X)
    y = np.asanyarray(y)
    return X, y
