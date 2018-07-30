import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# 계속해서 비디오를 읽어온다.
while True:
    ret, frame = cap.read()

    # frame 색을 gray색으로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('frame1', frame)  # imgshow
    cv2.imshow('gray', gray)
    cv2.imshow('blue', blue)
    cv2.imshow('frame4', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
