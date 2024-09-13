# Project-DL4CV
Mentor : Lê Nguyễn Thanh Huy ([CoTAI - Center of Talent in AI](https://gem.cot.ai/))\
Special thanks to  my mentor Lê Nguyễn Thanh Huy for your awesome lectures, sharing usefull information and experience with me.
## How to run the code
First you need to clone this repository to your local machine. Then open terminal and paste this command line
```
git clone https://github.com/hoangdangthien/Project-DL4CV.git
```
Then move into the clone directory
```
cd Project-DL4CV
```
Next download data from the following link:  [SH17dataset](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)\
Create folder ___datasets___ , then extract data and copy ___images___, ___labels___ and ___voc_labels___ folders to ___datasets___ folder\
Create a virtual environment with venv
```
python -m venv .venv
```
Activate the enviroment
```
.\.venv\Scripts\activate
```
Install all neccessary libraries with a specific version
```
pip install -r requirements.txt
```
To fine-tune model from ultralytics 
```
python train.py
```
To run streamlit app
```
streamlit run app.py
```
