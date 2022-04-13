import cv2

def main():

    backSub = cv2.createBackgroundSubtractorMOG2()
    #backSub = cv2.createBackgroundSubtractorKNN()

    capture = cv2.VideoCapture(1)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        fgMask = backSub.apply(frame)
        
        bgImg = backSub.getBackgroundImage()

        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
        
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        cv2.imshow('Background', bgImg)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()