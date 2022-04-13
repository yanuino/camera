import cv2
import numpy as np
from imutils import perspective


cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

# define range of color in HSV
lower = np.array([18,100,100])
upper = np.array([38,255,255])

while(True):
    ret, frame = cap.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    
    yellow = np.uint8([[upper]])  
    img = cv2.cvtColor(yellow, cv2.COLOR_HSV2BGR)
    (b, v, r) = img[0, 0]
    mask = cv2.inRange(im, lower, upper)
    
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows,cols = frame.shape[:2]

    for c in contours:
                
                # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 100:
            epsilon = 0.01*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(res,[approx],0,(int(b),int(v),int(r)), cv2.FILLED)
            
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(res,(x,y), (x+w,y+h), (255,0,0), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)

            box = perspective.order_points(box)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,0,255),2)
            
            # #  fit line
            # [vx,vy,x1,y1] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
            # # # make line perpendicular
            # # nx,ny = 1,-vx/vy
            # # mag = np.sqrt((1+ny**2))
            # # vx,vy = nx/mag,ny/mag
            # lefty = int((-x1*vy/vx) + y1)
            # righty = int(((cols-x1)*vy/vx)+y1)
            # cv2.line(frame,(cols-1,righty),(0,lefty),(0,255,0),1)

            # # angle
            # x_axis      = np.array([1, 0])    # unit vector in the same direction as the x axis
            # your_line   = np.array([vx, vy])  # unit vector in the same direction as your line
            # dot_product = np.dot(x_axis, your_line)
            # angle_2_x   = math.degrees(math.acos(dot_product))

            (tl, tr, br, bl) = box
            midr = br + (tr - br)//2
            midl = bl + (tl - bl)//2
            center, _, angle = rect
            center = np.int0(center)
            cv2.line(res, (x+w//2, y), (x+w//2, y+h), (0, 255, 0), 2)
            cv2.line(res, midr, midl, (0,0,255),1)
            # text = "Angle: " + str(round(angle)) + " degree"
            # cv2.putText(res, text, center + np.int0((w/2, h/2+14)), font, 0.5, (0,255,255), 1, cv2.LINE_AA)
            text = "Width: " + str(round(w)) + "pixels"
            # cv2.putText(res, text, center + (w//2 + 20, h//2 + 20), font, 0.5, (0,255,255), 1, cv2.LINE_AA)
            cv2.putText(res, text, (x + w, y + h), font, 0.5, (0,255,255), 1, cv2.LINE_AA)

           
    cv2.imshow('res', res)
    cv2.imshow('frame', frame)
    # break

# while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()