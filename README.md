# Project-DL4CV
Mentor : Lê Nguyễn Thanh Huy ([CoTAI - Center of Talent in AI](https://gem.cot.ai/))\
Special thanks to  my mentor Lê Nguyễn Thanh Huy for your awesome lectures, sharing useful information and experience with me.
## How to run the code
First you need to clone this repository to your local machine. Then open terminal and paste this command line
```
git clone https://github.com/hoangdangthien/Project-DL4CV.git
```
Then move into the clone directory
```
cd Project-DL4CV
```
Next download data from the following link:  [kaggle](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)\
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
To run error analysis
```
python 'Error analysis.py'
```
To run streamlit app
```
streamlit run app.py
```
Fine-tuned model's weights restore in folder ___models/YOLO/___ ,  demo video is in folder ___Video___
## Acknowledment
- The dataset is great. The original paper about dataset was written by Mughees Ahmad PhD Researcher and University of Windsor. You can [find more informatinon here](https://github.com/ahmadmughees/SH17dataset) and cite it using the following :
```
@article{ahmad_2024_sh17,
  title={SH17: A Dataset for Human Safety and Personal Protective Equipment Detection in Manufacturing Industry},
  author={Ahmad, Hafiz Mughees and Rahimi, Afshin},
  journal={Arxiv},
  year={2024}
}
```
-  Training of the models is done using [ultralytics](https://github.com/ultralytics/ultralytics) repository. Thanks for the great implementations.
