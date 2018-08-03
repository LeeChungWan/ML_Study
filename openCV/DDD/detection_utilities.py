import cv2
import numpy as np

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "haarcascade_eye.xml"
RED_COLOR = (0, 0, 255)

# Face Detection using Haar Cascades
# URL : https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
def getFaceCoordinates(img):
    # cv2.CascadeClassifier 특정 객체를 학습시켜 구분하기위함
    # 여기서는 사람 얼굴 검출하기 위함
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    # 회색으로 색 변환.
    imgToGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # frmae_img를 histogramEqualization 해준다.
    # 설명 URL : https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    src_img = cv2.equalizeHist(imgToGray)

    # CascadeClassifier의 detectMultiScale 함수에 grayscale 이미지를 입력하여 얼굴을 검출합니다.
    # 얼굴이 검출되면 위치를 리스트로 리턴합니다.
    # 이 위치는 (x, y, w, h)와 같은 튜플이며 (x, y)는 검출된 얼굴의 좌상단 위치
    # w, h는 가로, 세로 크기입니다.
    coordinates = cascade.detectMultiScale(
        image=src_img,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(48, 48))
    return coordinates


def drawFace(frame, face_coordinates):
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), RED_COLOR, thickness=2)
