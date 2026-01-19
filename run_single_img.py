from ultralytics import YOLO
import cv2
import os

# Load pretrained multi-person pose model
model = YOLO("yolov8n-pose.pt")  # n, s, m, l available

# Force GPU
model.to("cuda")

print ("Model loaded")

# Run inference
img = "000000000785.jpg"
#img = "imgs\\0105.jpg"

results = model(img, conf=0.25)

annotated = results[0].plot()

cv2.imwrite("output.jpg", annotated)
