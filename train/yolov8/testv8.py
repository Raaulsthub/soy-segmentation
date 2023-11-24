from ultralytics import YOLO

# Load a model
model = YOLO('../runs/yolov8x-seg/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='./data.yaml', device='cpu')  # no arguments needed, dataset and settings remembered

