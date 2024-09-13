import streamlit as st 
import cv2 
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import DeepOCSORT
import tempfile
from pathlib import Path

#load model
@st.cache_resource
def load_model():
    return YOLO('./models/YOLO/best.pt')

#load tracker 
@st.cache_resource
def load_tracker():
    return DeepOCSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'),
        device="cuda:0",
        fp16=True,
    )
#read video in frame, detect object in each frame, tracking only person, find which person the ppe belong to
def associate_ppe_with_persons(persons, ppe_items,CLASS_NAMES):
    person_ppe = {person[4]: set() for person in persons}
    
    if  len(persons)==0:
        return person_ppe

    person_centers = [[(p[0] + p[2]) / 2, (p[1] + p[3]) / 2] for p in persons]
    
    for ppe in ppe_items:
        distances=[]
        ppe_center = np.array([(ppe[0] + ppe[2]) / 2, (ppe[1] + ppe[3]) / 2])
        distances = np.linalg.norm(person_centers - ppe_center, axis=1)
        nearest_person_idx = np.argmin(distances)
        person_ppe[persons[nearest_person_idx][4]].add(CLASS_NAMES[int(ppe[4])])
    
    return person_ppe

def process_frame(frame,model,tracker,CLASS_NAMES,REQUIRED_PPE):
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
    
    person_ppe = associate_ppe_with_persons(tracks, ppe_detections,CLASS_NAMES)

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
    return frame
def reset_app_state():
    # Clear all st.session_state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Clear cache
    st.cache_resource.clear()
def main():
    st.title("PPE Tracking App")
    #init everything 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define class names (adjust these according to your model's classes)
    CLASS_NAMES = ["person","ear","ear-mufs","face","face-guard",'face-mask-medical',
                                "foot","tools","glasses","gloves","helmet","hands","head",
                                "medical-suit","shoes","safety-suit","safety-vest"]

    # Define the specific PPE items we want to detect
    REQUIRED_PPE = {"helmet", "safety-vest","shoes"}
    start = st.button("Start")
    if st.button('Stop'):
        reset_app_state()
        st.rerun()

    model = load_model()
    tracker = load_tracker()

    option = st.selectbox("Choose input source", ["Upload Video", "Use Webcam"])
    save_video = st.checkbox("Save processed video")

    if save_video:
        save_path = st.text_input("Enter the path to save the video", "./output1.mp4")

    if option == "Upload Video":
    
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None and start:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            vf = cv2.VideoCapture(tfile.name)
            
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vf.get(cv2.CAP_PROP_FPS))
            
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            stframe = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                frame = process_frame(frame, model, tracker, CLASS_NAMES,REQUIRED_PPE)
                stframe.image(frame, channels="BGR")
                
                if save_video:
                    out.write(frame)
            
            vf.release()
            if save_video:
                out.release()
                st.success(f"Video saved to {save_path}")

    elif option == "Use Webcam":
        cap = cv2.VideoCapture(-1,cv2.CAP_V4L)
        stframe = st.empty()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Assuming 30 fps for webcam

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        stop_button = st.button('Stop')

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = process_frame(frame, model, tracker, CLASS_NAMES,REQUIRED_PPE)
            stframe.image(frame, channels="BGR")
            
            if save_video:
                out.write(frame)

            stop_button = st.button('Stop')

        cap.release()
        if save_video:
            out.release()
            st.success(f"Video saved to {save_path}")

if __name__ == "__main__":
    main()
