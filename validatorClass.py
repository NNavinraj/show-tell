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
#configure db
db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] =  db['mysql_user']
app.config['MYSQL_PASSWORD'] =  db['mysql_password']
app.config['MYSQL_DB'] =  db['mysql_db']
 
mysql = MySQL(app)

validatorClass = Blueprint("validatorClass", __name__, static_folder="static", template_folder="templates")


def fileValidationCheck(fileType):
    import os
    import filetype
    import io
    from PIL import Image
    
    filetypess = fileType.filename.split('.')[-1]
    if fileType.filename == '':
        checkT = True
        checkTType = False
        return  "checkT"
    
    elif (filetypess != "jpg"):
        if(filetypess != "png"):
            checkT = False
            checkTType = True
            return  "checkTType"
        
        else:
            return "none"


def checkSqlConnection(db):
    import os
    import filetype
    import io
    from PIL import Image
    
    try:
        db.connection.cursor()
        return True
        
    except:
        return False

 


    