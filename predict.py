import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from cnn_preprocess import PreprocessCNN  # 👈 Custom CNN module

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO model
yolo_model = YOLO('./runs/detect/train/weights/last.pt')

# Load CNN model for preprocessing
preprocess_cnn = PreprocessCNN().to(device)
preprocess_cnn.eval()

# Input video
video_path = './videos/vid7.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# Output video
out = cv2.VideoWriter(
    f"{video_path}_out.mp4",
    cv2.VideoWriter_fourcc(*'avc1'),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (W, H)
)

threshold = 0.5

while ret:
    # Convert frame to RGB and Tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # CNN preprocessing
    with torch.no_grad():
        processed_tensor = preprocess_cnn(frame_tensor)
        processed_tensor = torch.clamp(processed_tensor, 0, 1)  # Ensure values are valid

    # Convert back to NumPy image for YOLO
    processed_img = (processed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

    # YOLO detection
    results = yolo_model(processed_img_bgr)[0]

    person_count = 0
    total_conf = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            person_count += 1
            total_conf += score
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = results.names[int(class_id)].upper()
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show count and average accuracy
    avg_conf = (total_conf / person_count) if person_count else 0
    info = f"Count: {person_count} | Accuracy: {avg_conf:.2f}"
    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
