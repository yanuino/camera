import pyperclip
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import re

cap = cv2.VideoCapture(2)
font = cv2.FONT_HERSHEY_SIMPLEX

oldbarCode = ''
newbarCode = ''

while(cap.isOpened()):
    ret, frame = cap.read()

    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
    decodedObjects = pyzbar.decode(im, symbols=[pyzbar.ZBarSymbol.QRCODE])

    # for decodedObject in decodedObjects: 
    if decodedObjects:
        decodedObject = decodedObjects[0]
    
        points = decodedObject.polygon
     
        # If the points do not form a quad, find convex hull
        if len(points) > 4 : 
          hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
          hull = list(map(tuple, np.squeeze(hull)))
        else : 
          hull = points;
         
        # Number of points in the convex hull
        n = len(hull)     
        # Draw the convext hull
        for j in range(0,n):
          cv2.line(frame, hull[j], hull[ (j+1) % n], (0,255,0), 2)

        rect = decodedObject.rect

        cv2.rectangle(frame, (rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height), (255,0,0), 1)
        # print('Type : ', decodedObject.type)
        # print('Data : ', decodedObject.data,'\n')

        newbarCode = decodedObject.data.decode('UTF-8')
        cv2.putText(frame, newbarCode, (2, 14), font, 0.5, (0,255,255), 1, cv2.LINE_AA)

    if newbarCode != oldbarCode:
        print(newbarCode)
        pyperclip.copy(newbarCode)
        oldbarCode = newbarCode

        expr = r"^CSN:(\w+);MAC:(\w+);.+;EAN code:(\w+)*"
        res = re.match(expr, newbarCode)
        if res:
            print(res[1])
            pyperclip.copy(res[1])
        

    # Display the resulting frame
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'): # wait for 's' key to save 
        cv2.imwrite('Capture.png', frame)     

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()