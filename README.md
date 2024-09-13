# Project-DL4CV
## How to run the code
First you need to clone this repository to your local machine. Then open terminal and paste this command line
```
git clone https://github.com/hoangdangthien/Project-DL4CV.git
```
Then move into the clone directory
```
cd Project-DL4CV
```
Then download data from the following link:\
[SH17dataset](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)\
Next extract data and copy all folder to datasets folder\
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
