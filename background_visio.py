import numpy as np
import cv2

def nothing(x):
    pass

def input_window():
    cv2.namedWindow('input', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Blur', 'input', 21, 240, nothing)
    cv2.createTrackbar('Canny_Low', 'input', 15, 240, nothing)
    cv2.createTrackbar('Canny_High', 'input', 150, 240, nothing)
    # cv2.createTrackbar('Min_Area', 'input', 0, 1000, nothing)
    # cv2.createTrackbar('Max_Area', 'input', 0, 1000, nothing)
    cv2.createTrackbar('Dilate_Iter', 'input', 10, 240, nothing)
    cv2.createTrackbar('Erode_Iter', 'input', 10, 240, nothing)
    
    blur = 21
    canny_low = 15
    canny_high = 150
    min_area = 0.0005
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10
    mask_color = (0.0,0.0,0.0)

def main():
    cap = cv2.VideoCapture(1)

    input_window()

    while True:
        ret, frame = cap.read()

        blur = cv2.getTrackbarPos('Blur', 'input')
        canny_low = cv2.getTrackbarPos('Canny_Low', 'input')
        canny_high = cv2.getTrackbarPos('Canny_High', 'input')
        min_area = 0.0005
        max_area = 0.95
        dilate_iter = cv2.getTrackbarPos('Dilate_Iter', 'input')
        erode_iter = cv2.getTrackbarPos('Erode_Iter', 'input')
        mask_color = (0.0,0.0,0.0)

        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image_gray, canny_low, canny_high)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        image_area = frame.shape[0] * frame.shape[1]  
        max_area = max_area * image_area
        min_area = min_area * image_area

        for c in contours[1]:
            if (cv2.contourArea(c)) > min_area:
                pass

        cv2.imshow("Out", edges)
        # contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]

        # image_area = frame.shape[0] * frame.shape[1]  
        # max_area = max_area * image_area
        # min_area = min_area * image_area

        # mask = np.zeros(edges.shape, dtype = np.uint8)

        # for contour in contour_info:
        #     if contour[1] > min_area and contour[1] < max_area:
        #         mask = cv2.fillConvexPoly(mask, contour[0], (255))

        # mask = cv2.dilate(mask, None, iterations=dilate_iter)
        # mask = cv2.erode(mask, None, iterations=erode_iter)
        # mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        # mask_stack = mask_stack.astype('float32') / 255.0           
        # frame = frame.astype('float32') / 255.0
        
        # masked = (mask_stack * frame) + ((1-mask_stack) * mask_color)
        # masked = (masked * 255).astype('uint8')
        
        # cv2.imshow("Foreground", masked)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  

if __name__ == "__main__":
    main()