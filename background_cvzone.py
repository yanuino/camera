import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np

def nothing(x):
    pass


segmentor = SelfiSegmentation()
img_b = cv2.imread('./bg/HomeStudy_03.jpg') # use it instead of (0,0,0) in removeBG
img_b=cv2.resize(img_b, [1280,720])
capture = cv2.VideoCapture(1)
cv2.namedWindow('Frame')
cv2.createTrackbar('Thr', 'Frame', 30, 100, nothing)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv2.CAP_PROP_FPS, 30)

with pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=PixelFormat.BGR) as cam:

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        thresh = cv2.getTrackbarPos('Thr', 'Frame')

        # frame = cv2.resize(frame,[512,512])

        out = segmentor.removeBG(frame, img_b, threshold=thresh/100)
        # out = segmentor.removeBG(frame, (0,0,0), threshold=thresh/100)
        cam.send(out)
        cam.sleep_until_next_frame()
        cv2.imshow('Frame', out)

        if cv2.waitKey(25) & 0xFF == ord('q'):
                break    
cv2.destroyAllWindows()
