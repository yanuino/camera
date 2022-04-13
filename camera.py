import cv2
from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
devices = enumerate(graph.get_input_devices())

for device in devices:
    print(f"{device}")



cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_SETTINGS, 1)

