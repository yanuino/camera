import cv2
import mediapipe as mp
mp_selfie_segmentation = mp.solutions.selfie_segmentation

import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np

def nothing(x):
    pass


# img_b = cv2.imread('./bg/HomeStudy_03.jpg') # use it instead of (0,0,0) in removeBG
# img_b = cv2.resize(img_b, [1280,720])
img_b = None
capture = cv2.VideoCapture(1)
cv2.namedWindow('Frame')
cv2.createTrackbar('Thr', 'Frame', 30, 100, nothing)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv2.CAP_PROP_FPS, 30)


with pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=PixelFormat.BGR) as cam:

    BG_COLOR = (192, 192, 192) # gray
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:

        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            thresh = cv2.getTrackbarPos('Thr', 'Frame')

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            
            frame.flags.writeable = False
            results = selfie_segmentation.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > thresh/100
            condition = np.dstack((results.segmentation_mask,) * 3) > thresh/100
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)
            black = np.zeros(frame.shape, dtype=np.uint8)
            black[:] = (0,0,0)
            white = np.zeros(frame.shape, dtype=np.uint8)
            white[:] = (255,255,255)
            img_c = np.where(condition, white, black)
            img_c = cv2.GaussianBlur(img_c, (55,55), 0)
            img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
            (_, img_c) = cv2.threshold(img_c, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # img_c = cv2.threshold(img_c, blur, 255, cv2.THRESH_BINARY)[1]
            img_c = np.dstack((img_c,)*3)

            # cv2.imshow('Cond', img_c)
            if img_b is None:
                img_b = np.zeros(frame.shape, dtype=np.uint8)
                img_b[:] = BG_COLOR
            img_b = cv2.GaussianBlur(frame,(55,55),0)
            out = np.where(img_c, frame, img_b)

            # out = segmentor.removeBG(frame, (0,0,0), threshold=thresh/100)
            cam.send(out)
            cam.sleep_until_next_frame()
            cv2.imshow('Frame', out)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                    break    
cv2.destroyAllWindows()
