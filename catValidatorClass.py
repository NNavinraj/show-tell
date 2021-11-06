from flask import Flask,Blueprint, render_template, Flask, redirect, url_for, render_template, request
import yaml
from flask_mysqldb import MySQL

import cv2
import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
import pyttsx3
from gtts import gTTS
from playsound import playsound
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from PIL import ImageFile   
from keras.preprocessing import image                  
from tqdm import tqdm
import ntpath

from flask_mysqldb import MySQL
import numpy as np
import re
import base64
from PIL import Image
from keras.models import load_model
import json
import yaml

app = Flask(__name__)
catClassifyClass = Blueprint("catClassifyClass", __name__, static_folder="static", template_folder="templates")
#configure db
db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] =  db['mysql_user']
app.config['MYSQL_PASSWORD'] =  db['mysql_password']
app.config['MYSQL_DB'] =  db['mysql_db']
 
mysql = MySQL(app)

#load the cat dataset
def load_dataset(path):
    data = load_files(path)
    cat_files = np.array(data['filenames'])
    cat_targets = np_utils.to_categorical(np.array(data['target']), 134)
    return cat_files, cat_targets

train_files, train_targets = load_dataset("static/CatIdentification/Train")
valid_files, valid_targets = load_dataset("static/CatIdentification/Valid")
test_files, test_targets = load_dataset("static/CatIdentification/Test")

cat_names = [item[20:-1] for item in sorted(glob("static/CatIdentification/Train/*/"))]

confthres = 0.3
nmsthres = 0.1
yolo_path = './'




@catClassifyClass.route("/catclassify", methods= ['POST'])
def catclassify():
    if request.method == "POST":
        import io
        from tensorflow.keras import backend as K
        from pickle import load
        K.clear_session()
        
        return render_template("catclassify.html")
        
 
    
@catClassifyClass.route("/catclassify", methods= ['GET'])
def viewCat():
    if request.method == "GET":
        return render_template("/catclassify.html")
 

class Cat:

    def __init__(self,name):
        self.name = name
        pass    # instance variable unique to each instance
        
    def getName(self):
        return self.name
    