import fiftyone as fo 
import random 
import cv2 
from matplotlib import pyplot as plt 
import albumentations as A 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor,AutoModelForObjectDetection, TrainingArguments,Trainer
import torch
import numpy as np
import xml.etree.ElementTree as ET
import os
from sklearn.model_selection import train_test_split

#get list name of files in train,test,val
abs_path = os.path.dirname(__file__)
with open(abs_path+"/datasets/train.txt") as f:
    train_path = [line.rstrip() for line in f ]

with open(abs_path+"/datasets/test.txt") as f:
    test_path = [line.rstrip() for line in f ]

with open(abs_path+"/datasets/val.txt") as f:
    val_path = [line.rstrip() for line in f ]

labelname = ["person","ear","ear-mufs","face","face-guard",'face-mask-medical',"foot","tools","glasses","gloves","helmet","hands","head","medical-suit","shoes","safety-suit","safety-vest"]
id2label = {i : label for i,label in enumerate(labelname)}
label2id = {label : i for i,label in enumerate(labelname)}

label_path = abs_path+'/datasets/labels/'
samples=[]
def read_anno(file):
    with open(file,'r') as f:
        lines = [line.rstrip() for line in f]
        list_obj=[]
        for line in lines:
            line = line.split(" ")
            line[0] = int(line[0])
            line[1] = float(line[1])
            line[2] = float(line[2])
            line[3] = float(line[3])
            line[4] = float(line[4])
            line[1] = line[1]-line[3]/2
            line[2] = line[2]-line[4]/2
            list_obj.append(line)
    return list_obj
for file in test_path:
    img_name = file.split(".")[0].split("/")[-1]
    sample = fo.Sample(filepath=file)
    detections=[]
    list_obj = read_anno(label_path+img_name+'.txt')
    for obj in list_obj:
        ids = obj[0]
        label = id2label[ids]
        bounding_box = obj[1:]
        detections.append(
            fo.Detection(label=label,bounding_box=bounding_box)
        )
    sample["ground_truth"] = fo.Detections(detections=detections)

    samples.append(sample)
dataset = fo.Dataset("dataset111")
dataset.add_samples(samples)


#dataset = fo.Dataset("my-dataset")
#dataset.add_samples(samples)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
import fiftyone.utils.yolo as fouy
from torchvision.transforms import functional as func


sample = dataset.first()
from ultralytics import YOLO,RTDETR
model = YOLO(abs_path+'/models/YOLO/best.pt')
#model = RTDETR("/mnt/d/DL4CV/Myproject/test/runs/train2/weights/best.pt")
predictions_view = dataset.take(100,seed=51)

from PIL import Image

import fiftyone as fo

# Get class list
classes = dataset.default_classes

# Add predictions to samples
with fo.ProgressBar() as pb:
    for sample in pb(predictions_view):
        # Load image
        image = np.array(Image.open(sample.filepath).convert("RGB"))
        image = torch.tensor(image).to(device)
        h,w,c = image.shape

        # Perform inference
        preds = model.predict(sample.filepath,conf=0.5)
        labels = preds[0].boxes.cls.cpu().numpy()
        print(labels)
        scores = preds[0].boxes.conf.cpu().numpy()
        #print(score)
        boxes = preds[0].boxes.xyxy.cpu().numpy()
        #print(boxes)

        # Convert detections to FiftyOne format
        detections = []
        for label, score, box in zip(labels, scores, boxes):
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x1, y1, x2, y2 = box
            rel_box = [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h]

            detections.append(
                fo.Detection(
                    label=id2label[int(label)],
                    bounding_box=rel_box,
                    confidence=score
                )
            )

        # Save predictions to dataset
        #print(detections)
        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

session = fo.launch_app(dataset)
session.view = predictions_view