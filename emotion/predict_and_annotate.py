import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from emotion_cnn import EmotionCNN  # make sure this matches your model file
import os

# --------------- Step 1: Load Models ---------------

# Load YOLOv8 face detection model
face_model = YOLO("yolov8n-face.pt")

# Load emotion classification model
emotion_model = EmotionCNN()
emotion_model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
emotion_model.eval()

# --------------- Step 2: Define Transforms ---------------

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --------------- Step 3: Define Emotion Labels ---------------

emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# --------------- Step 4: Predict Emotions on Faces ---------------

def predict_emotions_on_faces(image_path, output_path="annotated_output.jpg"):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_model(image_rgb)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords
            face_crop = image_rgb[y1:y2, x1:x2]

            # Handle edge case where box is out of image bounds
            if face_crop.size == 0 or x2 <= x1 or y2 <= y1:
                continue

            # Convert face to PIL Image and preprocess
            face_pil = Image.fromarray(face_crop)
            face_tensor = transform(face_pil).unsqueeze(0)  # Add batch dim

            # Predict emotion
            with torch.no_grad():
                outputs = emotion_model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                label = emotion_labels[predicted.item()]

            # Annotate image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save output
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

# --------------- Step 5: Run ---------------

if __name__ == "__main__":
    predict_emotions_on_faces("crowd.jpg")
