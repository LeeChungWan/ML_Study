import cv2
import sys
import time
import detection_utilities as du

windowName = 'Webcam Screen'

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break


def getCameraStreaming():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming")
        sys.exit()
    print("Successed to capture video streaming")
    return capture


def setDefaultCameraSetting():
    cv2.startWindowThread()
    cv2.namedWindow(winname=windowName)
    cv2.setWindowProperty(winname=windowName, prop_id=cv2.WINDOW_FULLSCREEN, prop_value=cv2.WINDOW_FULLSCREEN)


def showScreenAndDetectFace(capture):
    while True:
        ret, frame = capture.read()
        face_coordinates = du.getFaceCoordinates(frame)
        refreshScreen(frame, face_coordinates)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


def refreshScreen(frame, face_coordinates):
    if face_coordinates is not None:
        cropped = du.crop_face(frame, face_coordinates)
        cv2.imshow('crop', cropped)
        du.drawFace(frame, face_coordinates)
    cv2.imshow(windowName, frame)

def main():
    print("Start main() function.")

    capture = getCameraStreaming()
    setDefaultCameraSetting()
    showScreenAndDetectFace(capture)


if __name__ == '__main__':
    main()

# cap.release()
cv2.destroyAllWindows()
