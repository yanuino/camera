import numpy as np
import cv2

def nothing(x):
    pass

def decode1(image, treshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    ret, thresh = cv2.threshold(blurred, treshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded", thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", image)

def decode(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # equalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # edge enhancement
    edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, 
                            ksize = 3, scale = 1, delta = 0)
    cv2.imshow("Edges", edge_enh)
    #cv2.waitKey(0)
    #retval = cv2.imwrite("edge_enh.jpg", edge_enh)

    # bilateral blur, which keeps edges
    blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)

    # use simple thresholding. adaptive thresholding might be more robust
    ret, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded", thresh)
    #cv2.waitKey(0)
    #retval = cv2.imwrite("thresh.jpg", thresh)

    # do some morphology to isolate just the barcode blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    cv2.imshow("After morphology", closed)
    #cv2.waitKey(0)
    #retval = cv2.imwrite("closed.jpg", closed)

    # find contours left in the image
    cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    print(box)
    cv2.imshow("found barcode", image)
    #cv2.waitKey(0)
    #retval = cv2.imwrite("found.jpg", image)

def main():
    cap = cv2.VideoCapture(1)

    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('input')
    cv2.createTrackbar('Thr', 'input', 0, 240, nothing)

    while(True):
        ret, frame = cap.read()
        thresh = cv2.getTrackbarPos('Thr', 'input')
        decode(frame, thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()