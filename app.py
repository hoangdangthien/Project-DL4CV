import cv2
import torch
import numpy as np
from ultralytics import RTDETR, YOLO
from boxmot import DeepOCSORT
from pathlib import Path
import os

#get path
abs_path = os.path.dirname(__file__)

# Initialize YOLO model
#model = RTDETR('./models/RTDETR/best.pt')
model = YOLO(abs_path+'/models/YOLO/best.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define class names (adjust these according to your model's classes)
CLASS_NAMES = ["person","ear","ear-mufs","face","face-guard",'face-mask-medical',
                            "foot","tools","glasses","gloves","helmet","hands","head",
                            "medical-suit","shoes","safety-suit","safety-vest"]

# Define the specific PPE items we want to detect
REQUIRED_PPE = {"helmet", "safety-vest","shoes"}

# Initialize DeepOCSORT tracker
tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'),  # path to DeepOCSORT weights if you have them
    device=device,  # use "cuda" if you have a GPU
    fp16=False
)

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

# Initialize color map for classes
color_map = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3), dtype=np.uint8)

cap = cv2.VideoCapture(abs_path+"/Video/Take Time to Take Care (Working at Heights) Video.mp4")  # Use 0 for webcam or provide video file path
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(abs_path+'/Video/output_video.mp4', fourcc, fps, (width, height))
def associate_ppe_with_workers(workers, ppe_items):
    worker_ppe = {worker[4]: set() for worker in workers}
    
    if  len(workers)==0:
        return worker_ppe

    worker_centers = [[(w[0] + w[2]) / 2, (w[1] + w[3]) / 2] for w in workers]
    
    for ppe in ppe_items:
        distances=[]
        ppe_center = np.array([(ppe[0] + ppe[2]) / 2, (ppe[1] + ppe[3]) / 2])
        distances = np.linalg.norm(worker_centers - ppe_center, axis=1)
        nearest_worker_idx = np.argmin(distances)
        worker_ppe[workers[nearest_worker_idx][4]].add(CLASS_NAMES[int(ppe[4])])
    
    return worker_ppe
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform RTDETR detection
    results = model.predict(frame,conf=0.3)

    # Extract bounding boxes, classes, and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    # Prepare detections for tracking (only person)
    person_detections = []
    ppe_detections = []
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        if CLASS_NAMES[int(cls)] == "person":
            person_detections.append([x1, y1, x2, y2, score, cls])
        elif CLASS_NAMES[int(cls)] in REQUIRED_PPE:
            ppe_detections.append([x1, y1, x2, y2,cls])

    # Update tracker with worker detections only
    if len(person_detections)>0:
        tracks = tracker.update(np.array(person_detections), frame)
    else:
        person_detections=np.empty((0,6))
        tracks = tracker.update(np.array(person_detections), frame)
    
    person_ppe = associate_ppe_with_workers(tracks, ppe_detections)

    # Process and visualize results
    for track in tracks:
        bbox = track[:4]
        track_id = int(track[4])

        x1, y1, x2, y2 = map(int, bbox)
        
        # Determine if worker has all required PPE
        has_all_ppe = REQUIRED_PPE.issubset(person_ppe[track_id])
        
        # Draw bounding box
        color = (0, 255, 0) if has_all_ppe else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"Worker {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display specific PPE status
        y_offset = y2 + 20
        for ppe_item in REQUIRED_PPE:
            status = "yes" if ppe_item in person_ppe[track_id] else "no"
            color_ppe = (0,255,0) if status=="yes" else (0,0,255)
            cv2.putText(frame, f"{ppe_item}: {status}", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ppe, 2)
            y_offset += 20

    # Display the frame
    cv2.imshow('Worker Tracking with PPE Detection', frame)
    #write frame to output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()