import cv2
from time import sleep

CAM_ID = 0


def capture(camid=CAM_ID):
    cam = cv2.VideoCapture(camid)
    if cam.isOpened() == False:
        print('cant open the cam (%d)' % camid)
        return None
    # range(N) N으로 사진 갯수 조정
    for i in range(20):
        file_name = str(i) + '.png'
        file_path = './data/' + file_name
        ret, frame = cam.read()
        if frame is None:
            print('frame is not exist')
            return None
        cv2.imwrite(file_path, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        # sleep(N) N으로 시간 간격으로 사진을 찍는다.
        sleep(2)
    cam.release()


if __name__ == '__main__':
    capture()
