from ultralytics import YOLO
import cv2
import os

# Create output folder
os.makedirs("crops", exist_ok=True)

# Load YOLOv8 face model
model = YOLO("yolov8n-face.pt")

# Read crowd image
img_path = "crowd.jpg"
image = cv2.imread(img_path)

# Run detection
results = model(image)

# Crop and save faces
for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    face = image[y1:y2, x1:x2]
    cv2.imwrite(f"crops/face_{i}.jpg", face)

print("✅ Faces cropped and saved to 'crops/' folder.")
