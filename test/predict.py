import cv2
from ultralytics import YOLO

model = YOLO('../train/runs/yolov8x-seg/weights/best.pt')

image = cv2.imread('./images/img4.jpg')
results = model.predict(source=image, save=True, save_txt=True, conf=0.2, show=False, boxes=True)
