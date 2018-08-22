import cv2

cap = cv2.VideoCapture(0)
window_name = "frame"


def getCoordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)


def main():
    while True:
        ret, frame = cap.read()
        cv2.setMouseCallback(window_name, getCoordinate)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
