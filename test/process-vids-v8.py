import cv2
from ultralytics import YOLO

model = YOLO('../train/runs/yolov5x-seg/weights/best.pt')
cap = cv2.VideoCapture("../image-extraction/videos/20230114-GX010238.MP4")

output_video = "./videos/yolov5x/20230114-GX010238NBB.mp4"  # Output video file name with MP4 format
image_path = './runs/segment/predict/image0.jpg'
frame_counter = 0

ret, frame = cap.read()

frame = cv2.resize(frame, (640, 640))

results = model.predict(source=frame, save=True, save_txt=True, conf=0.2, show=False, boxes=False)

img = cv2.imread(image_path)

# get img properties
height, width, layers = img.shape
# get fps from cap
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))

    results = model.predict(source=frame, save=True, save_txt=True, conf=0.2, show=False, boxes=False)

    # Add the frame to the video
    img = cv2.imread(image_path)
    cv2.imshow('predict', img)
    video.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cap.release()
cv2.destroyAllWindows()
