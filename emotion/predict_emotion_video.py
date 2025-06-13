import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from emotion_cnn import EmotionCNN  # Make sure this matches your file

# ---------- Step 1: Load Models ----------

# Load YOLOv8 face detector
face_model = YOLO("yolov8n-face.pt")

# Load Emotion Classifier
emotion_model = EmotionCNN()
emotion_model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device("cpu")))
emotion_model.eval()

# Emotion labels
emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Preprocessing for emotion model
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------- Step 2: Emotion Prediction on Frame ----------

def annotate_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_model(image_rgb)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = image_rgb[y1:y2, x1:x2]

            if face_crop.size == 0 or x2 <= x1 or y2 <= y1:
                continue

            face_pil = Image.fromarray(face_crop)
            face_tensor = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = emotion_model(face_tensor)
                _, predicted = torch.max(outputs, 1)
                label = emotion_labels[predicted.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# ---------- Step 3: Video Processing ----------

def process_video(input_path="input_video.mp4", output_path="output_video.mp4"):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = annotate_frame(frame)
        out.write(frame)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Done! Annotated video saved as {output_path}")

# ---------- Step 4: Run Script ----------

if __name__ == "__main__":
    process_video("input_video.mp4", "output_video.mp4")
