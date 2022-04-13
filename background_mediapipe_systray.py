import cv2
import mediapipe as mp
mp_selfie_segmentation = mp.solutions.selfie_segmentation

import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np

import sys, os
from pystray import Icon as icon, MenuItem as item, Menu as menu
from PIL import Image

from pygrabber.dshow_graph import FilterGraph

import threading

state = 0
running = False
mutex = threading.Lock()
stop_thread = False

def nothing(x):
    pass

def genlist():
    return ( item('{0}'.format(desc), set_state(i), visible=True, default=False, checked=get_state(i)) for i, desc in get_devices() )

def get_devices():
    graph = FilterGraph()
    return ((id, desc) for id, desc in enumerate(graph.get_input_devices()) if desc != "Unity Video Capture")

def on_clicked(icon, item):
    global running
    global mutex
    running = not item.checked
    if running:
        mutex.release()
    else:
        mutex.acquire()

    #TODO: create a thread /destroy thread based on 'running' status
    #NOTE: evaluate if the thread should be here or launch in global way and reading 'running' value

def stop(icon):
    global stop_thread
    global mutex
    mutex.release()
    stop_thread = True
    icon.stop()

def set_state(v):
    def inner(icon, item):
        global state
        state = v
    return inner

def get_state(v):
    def inner(item):
        return state == v
    return inner

def pipe_blurbg(cameraId):
    # img_b = cv2.imread('./bg/HomeStudy_03.jpg') # use it instead of (0,0,0) in removeBG
    # img_b = cv2.resize(img_b, [1280,720])
    img_b = None
    capture = cv2.VideoCapture(cameraId)
    # cv2.namedWindow('Frame')
    # cv2.createTrackbar('Thr', 'Frame', 30, 100, nothing)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 30)


    with pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=PixelFormat.BGR) as cam:

        BG_COLOR = (192, 192, 192) # gray
        with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1) as selfie_segmentation:

            while True:
                mutex.acquire()
                mutex.release()
                ret, frame = capture.read()
                if frame is None:
                    break
                # thresh = cv2.getTrackbarPos('Thr', 'Frame')
                thresh = 30

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
                # cv2.imshow('Frame', out)
                global stop_thread
                if stop_thread:
                    break
    
    capture.release()
    cv2.destroyAllWindows()

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

mutex.acquire()
pipe = threading.Thread(target=pipe_blurbg, args=(1,))
pipe.start()

myimage = Image.open(os.path.join(application_path, 'Webcam_Icon.png'))

submenu = menu(genlist)
mymenu = menu(item('Pipe', on_clicked, checked=lambda item: running), item ('Quit', lambda icon: stop(icon)), item('Camera', submenu))

icon('test', myimage, menu=mymenu).run()
print('Test')


