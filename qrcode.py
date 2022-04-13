import cv2
import numpy as np
import sys
import time

def display_box(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[(j+1)%n][0]), (255,0,0),3)

    cv2.imshow('result', im)

vid = cv2.VideoCapture(2)
qr = cv2.QRCodeDetector()

while(True):
    ret, frame = vid.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    data,bbox,rectifiedImage = qr.detectAndDecode(im)

    if len(data)>0:
        #display_box(im, bbox)
        cv2.imshow('qrcode', rectifiedImage)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()