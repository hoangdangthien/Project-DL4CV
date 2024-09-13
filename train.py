import glob
import os
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

abs_path = os.path.dirname(__file__)
list_file = glob.glob(abs_path+"/datasets/images/"+"*.jpeg")
train,val = train_test_split(list_file,test_size=0.3)
val,test = train_test_split(val,test_size=0.5)
# create list txt file
with open(abs_path+"/datasets/train.txt",'w') as f:
    data = "\n".join(train)
    f.write(data)
with open(abs_path+"/datasets/val.txt",'w') as f:
    data = "\n".join(val)
    f.write(data)
with open(abs_path+"/datasets/test.txt",'w') as f:
    data = "\n".join(test)
    f.write(data)

#train model
model = YOLO(r"yolov8n.pt")
model.train(data="sh17.yaml",amp=False,batch=4,imgsz=640,optimizer='AdamW',lrf=0.0001,epochs=5)