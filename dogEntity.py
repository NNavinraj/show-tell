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

from validatorClass import fileValidationCheck
from validatorClass import checkSqlConnection

app = Flask(__name__)
dogEntity = Blueprint("dogEntity", __name__, static_folder="static", template_folder="templates")
#configure db
db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] =  db['mysql_user']
app.config['MYSQL_PASSWORD'] =  db['mysql_password']
app.config['MYSQL_DB'] =  db['mysql_db']
 
mysql = MySQL(app)


@dogEntity.route("/dogs")
def dogs():


    #insert sql statement to get names of dogs (seperated by breed size/HDB approved)
 
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select Breed from dog where HDB = 'HDB'"
        value = cur.execute(sql)
        hdb = cur.fetchall()
        sql = "select Breed from dog where Size = 'small'"
        value = cur.execute(sql)
        small = cur.fetchall()
        sql = "select Breed from dog where Size = 'Large'"
        value = cur.execute(sql)
        large = cur.fetchall()
        #to include the values
        return render_template("dogs.html", hdb=hdb, small=small, large=large)
    else:
        return render_template("sqlerror.html")
    cur.close()

    
@dogEntity.route("/dogs/<name>")
def dogbreed(name):
  
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select Breed, AverageLifeSpan, Size, Description, Characteristic from dog where Breed = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        return render_template("dogbreed.html", result=result, img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
    

@dogEntity.route("/dogs/buyheredog/<name>")
def buydog(name):
  
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = " select IC, petstoreanimal.petStoreID,name,DateOfBirth,gender,vaccindated,breed,price,size,hdb, address, telephone, email from petstoreanimal join petstore on petstoreanimal.petstoreID = petstore.petStoreID where Breed = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        return render_template("buyheredog.html", result=result, img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()
    


class Dog:

    def __init__(self,name):
        self.name = name
        pass    # instance variable unique to each instance
        
    def getName(self):
        return self.name
    