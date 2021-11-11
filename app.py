from flask import Flask, redirect, url_for, render_template, request
from flask_mysqldb import MySQL
import numpy as np
import re
import base64
from PIL import Image
from keras.models import load_model
import json
import yaml

# IMPORT CLASSES/ENTITIES OUTSIDE OF APP
from catClass import catClass
from catClass import Cat

from dogClass import dogClass
from dogClass import Dog

from catClassifyClass import catClassifyClass
from dogClassifyClass import dogClassifyClass

from validatorClass import validatorClass
from validatorClass import checkSqlConnection
import os
import filetype
import io
from PIL import Image

app = Flask(__name__)
app.register_blueprint(catClass, static_folder='../static')
app.register_blueprint(catClassifyClass, static_folder='../static')

app.register_blueprint(dogClass, static_folder='../static')
app.register_blueprint(dogClassifyClass, static_folder='../static')

app.register_blueprint(validatorClass, static_folder='../static')

#configure db
db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] =  db['mysql_user']
app.config['MYSQL_PASSWORD'] =  db['mysql_password']
app.config['MYSQL_DB'] =  db['mysql_db']
 
mysql = MySQL(app)

conv = load_model("./model/animals.h5")
ANIMALS = {0: "Bird", 1: "Cat", 2: "Dog", 3: "Rabbit"}

def normalize(data):
    "Takes a list or a list of lists and returns its normalized form"

    return np.interp(data, [0, 255], [-1, 1])

@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/navbar.html")
def navBarNew():
    return render_template("navbar.html")

@app.route("/footer.html")
def footerNew():
    return render_template("footer.html")
	
@app.route("/navbarfooter.html")
def navBarFooterNew():
    return render_template("navbarfooter.html")

@app.route("/sqlerror.html")
def sqlError():
    return render_template("sqlerror.html")

#DOG CLASSIFICATION CODE
#IMPORTS
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



#SG PET STORE INDEX 
@app.route("/sgPetStoreIndex.html")
def sgPetStoreHome():
    if (checkSqlConnection(mysql) == True):
        #insert sql statement to get names of cats
        cur = mysql.connection.cursor()
        sql = "select * from petstore where petStoreId = 'SG Pet Store'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("sgPetStoreIndex.html",cat=cat)
    else:
        return render_template("sqlerror.html")
    cur.close()



@app.route("/sgPetStoreContactUs.html")
def sgPetStoreContactUs():
    #insert sql statement to get names of cats
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select * from petstore where petStoreId = 'SG Pet Store'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("sgPetStoreContactUs.html",cat=cat)
    else:
        return render_template("sqlerror.html")
    cur.close()

@app.route("/PLCIndex.html")
def PLCHome():
    #insert sql statement to get names of cats
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select * from petstore where petStoreId = 'Pet Lovers Center'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("PLCIndex.html",cat=cat)
    else:
        return render_template("sqlerror.html")
    cur.close()

@app.route("/PLCContactUs.html")
def PLCContactUs():
    #insert sql statement to get names of cats
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select * from petstore where petStoreId = 'Pet Lovers Center'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("PLCContactUs.html",cat=cat)
    else:
        return render_template("sqlerror.html")
    
    cur.close()

@app.route("/PawsShopIndex.html")
def PawsShopHome():
    #insert sql statement to get names of cats
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select * from petstore where petStoreId = 'Paws Shop'"
        value = cur.execute(sql)
        cat = cur.fetchall()
    #to include the values
        return render_template("PawsShopIndex.html",cat=cat)
    else:
        return render_template("sqlerror.html")
    cur.close()

@app.route("/PawsShopContactUs.html")
def PawsShopContactUs():
    #insert sql statement to get names of cats
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select * from petstore where petStoreId = 'Paws Shop'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("PawsShopContactUs.html",cat=cat)
    else:
        return render_template("sqlerror.html")
    cur.close()

    

	
@app.route("/sgPetStoreDogs.html")
def sgPetStoreDogs():
    if (checkSqlConnection(mysql) == True):
        #insert sql statement to get names of dogs (seperated by breed size/HDB approved)
        cur = mysql.connection.cursor()
        sql = "select * from petStoreAnimal where HDB = 'HDB' and petStoreId = 'SG Pet Store'"
        value = cur.execute(sql)
        hdb = cur.fetchall()
        sql = "select * from petStoreAnimal where Size = 'small' and petStoreId = 'SG Pet Store'"
        value = cur.execute(sql)
        small = cur.fetchall()
        sql = "select * from petStoreAnimal where Size = 'Large' and petStoreId = 'SG Pet Store'"
        value = cur.execute(sql)
        large = cur.fetchall()
    
        #to include the values
        return render_template("sgPetStoreDogs.html", hdb=hdb, small=small, large=large)
    else:
        return render_template("sqlerror.html")
    cur.close()

@app.route("/PLCDogs.html")
def PLCDogs():
    
    if (checkSqlConnection(mysql) == True):
        #insert sql statement to get names of dogs (seperated by breed size/HDB approved)
        cur = mysql.connection.cursor()
        sql = "select * from petStoreAnimal where HDB = 'HDB' and petStoreId = 'Pet Lovers Center'"
        value = cur.execute(sql)
        hdb = cur.fetchall()
        sql = "select * from petStoreAnimal where Size = 'small' and petStoreId = 'Pet Lovers Center'"
        value = cur.execute(sql)
        small = cur.fetchall()
        sql = "select * from petStoreAnimal where Size = 'Large' and petStoreId = 'Pet Lovers Center'"
        value = cur.execute(sql)
        large = cur.fetchall()
    
        #to include the values
        return render_template("PLCDogs.html", hdb=hdb, small=small, large=large)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/PawsShopDogs.html")
def PawsShopDogs():
    if (checkSqlConnection(mysql) == True):
        #insert sql statement to get names of dogs (seperated by breed size/HDB approved)
        cur = mysql.connection.cursor()
        sql = "select * from petStoreAnimal where HDB = 'HDB' and petStoreId = 'Paws Shop'"
        value = cur.execute(sql)
        hdb = cur.fetchall()
        sql = "select * from petStoreAnimal where Size = 'small' and petStoreId = 'Paws Shop'"
        value = cur.execute(sql)
        small = cur.fetchall()
        sql = "select * from petStoreAnimal where Size = 'Large' and petStoreId = 'Paws Shop'"
        value = cur.execute(sql)
        large = cur.fetchall()
    
        #to include the values
        return render_template("PawsShopDogs.html", hdb=hdb, small=small, large=large)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/sgPetStoreCats.html")
def sgPetStoreCats():
    if (checkSqlConnection(mysql) == True):
        #insert sql statement to get names of cats
        cur = mysql.connection.cursor()
        sql = "select * from petStoreAnimal where petType ='cat' and petStoreId = 'SG Pet Store'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("sgPetStoreCats.html", cat=cat)

    else:
        return render_template("sqlerror.html")
    cur.close()

@app.route("/PLCCats.html")
def PLCCats():
    if (checkSqlConnection(mysql) == True):
        #insert sql statement to get names of cats
        cur = mysql.connection.cursor()
        sql = "select * from petStoreAnimal where petType ='cat' and petStoreId = 'Pet Lovers Center'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("PLCCats.html", cat=cat)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/PawsShopCats.html")
def PawsShopCats():
    if (checkSqlConnection(mysql) == True):
        #insert sql statement to get names of cats
        cur = mysql.connection.cursor()
        sql = "select * from petStoreAnimal where petType ='cat' and petStoreId = 'Paws Shop'"
        value = cur.execute(sql)
        cat = cur.fetchall()
        #to include the values
        return render_template("PawsShopCats.html", cat=cat)
    else:
        return render_template("sqlerror.html")
    cur.close()

@app.route("/SG Pet Store/buydog/<name>")
def sgPetStoreDogBreedBuy(name):
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'SG Pet Store'  and name = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='SG Pet Store' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
        value = cur.execute(sql)
        others = cur.fetchall()
        return render_template("sgPetStoreDogBreedBuy.html", result=result, others=others,img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/Pet Lovers Center/buydog/<name>")
def PLCDogBreedBuy(name):
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Pet Lovers Center'  and name = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Pet Lovers Center' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
        value = cur.execute(sql)
        others = cur.fetchall()
        return render_template("PLCDogBreedBuy.html", result=result, others=others,img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/Paws Shop/buydog/<name>")
def PawsShopDogBreedBuy(name):
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Paws Shop'  and name = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Paws Shop' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
        value = cur.execute(sql)
        others = cur.fetchall()
        return render_template("PawsShopDogBreedBuy.html", result=result, others=others,img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/SG Pet Store/buycat/<name>")
def sgPetStoreCatBreedBuy(name):
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'SG Pet Store'  and name = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='SG Pet Store' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
        value = cur.execute(sql)
        others = cur.fetchall()
        return render_template("sgPetStoreCatBreedBuy.html", result=result,others=others, img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/Pet Lovers Center/buycat/<name>")
def PLCCatBreedBuy(name):
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Pet Lovers Center'  and name = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Pet Lovers Center' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
        value = cur.execute(sql)
        others = cur.fetchall()
        return render_template("PLCCatBreedBuy.html", result=result,others=others, img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()
    
@app.route("/Paws Shop/buycat/<name>")
def PawsShopCatBreedBuy(name):
    if (checkSqlConnection(mysql) == True):
        cur = mysql.connection.cursor()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Paws Shop'  and name = '" + name + "'"
        value = cur.execute(sql)
        result = cur.fetchall()
        sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Paws Shop' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
        value = cur.execute(sql)
        others = cur.fetchall()
        return render_template("PawsShopCatBreedBuy.html", result=result,others=others, img=name)
    else:
        return render_template("sqlerror.html")
    cur.close()



if __name__ =="__main__":
   

	#app.debug = True
	app.run()
    