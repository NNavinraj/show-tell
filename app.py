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
from catEntity import catEntity
from catEntity import Cat

from dogEntity import dogEntity
from dogEntity import Dog

from catClassifyClass import catClassifyClass
from dogClassifyClass import dogClassifyClass

import os


app = Flask(__name__)
app.register_blueprint(catEntity, static_folder='../static')
app.register_blueprint(catClassifyClass, static_folder='../static')

app.register_blueprint(dogEntity, static_folder='../static')
app.register_blueprint(dogClassifyClass, static_folder='../static')


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


confthres = 0.3
nmsthres = 0.1
yolo_path = './'


def getPrediction(filename):
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import decode_predictions

    model1 = VGG16()
    image = load_img('static/temp.jpg', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model1.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    return label[1], label[2]*100




def get_labels(labels_path):
    import os
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    import os
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    import os
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_yolov3model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def image_to_byte_array(image:Image):
    import io
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

#from flask_bootstrap import Bootstrap
#from flask_ngrok import run_with_ngrok

labelsPath="weights/coco.names"
cfgpath="weights/yolov3.cfg"
wpath="weights/yolov3.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_yolov3model(CFG,Weights)
Colors=get_colors(Lables)

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

#path to tensor
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

#rescale images by 255
ImageFile.LOAD_TRUNCATED_IMAGES = True      
           
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255







@app.route("/doodle.html", methods=["GET", "POST"])
def doodle():
    if request.method == "GET":
        return render_template("doodle.html")
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        img = base64.decodebytes(data.encode('utf-8'))
        with open('temp.png', 'wb') as output:
            output.write(img)
        x = Image.open('temp.png').convert('L')
        # resize input image to 28x28
        x = x.resize((28, 28))
        model = conv
        x = np.expand_dims(x, axis=0)
        x = np.reshape(x, (28, 28, 1))
        # invert the colors
        x = np.invert(x)
        # brighten the image by 60%
        for i in range(len(x)):
            for j in range(len(x)):
                if x[i][j] > 50:
                    x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        pred = ANIMALS[np.argmax(val)]
        classes = ["Bird", "Cat", "Dog", "Rabbit"]

        return render_template("doodle.html", preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"],pred=pred)

#SG PET STORE INDEX 
@app.route("/sgPetStoreIndex.html")
def sgPetStoreHome():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petstore where petStoreId = 'SG Pet Store'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("sgPetStoreIndex.html",cat=cat)
    cur.close()


@app.route("/sgPetStoreContactUs.html")
def sgPetStoreContactUs():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petstore where petStoreId = 'SG Pet Store'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("sgPetStoreContactUs.html",cat=cat)
    cur.close()

@app.route("/PLCIndex.html")
def PLCHome():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petstore where petStoreId = 'Pet Lovers Center'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("PLCIndex.html",cat=cat)
    cur.close()

@app.route("/PLCContactUs.html")
def PLCContactUs():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petstore where petStoreId = 'Pet Lovers Center'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("PLCContactUs.html",cat=cat)
    cur.close()

@app.route("/PawsShopIndex.html")
def PawsShopHome():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petstore where petStoreId = 'Paws Shop'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("PawsShopIndex.html",cat=cat)
    cur.close()

@app.route("/PawsShopContactUs.html")
def PawsShopContactUs():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petstore where petStoreId = 'Paws Shop'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("PawsShopContactUs.html",cat=cat)
    cur.close()

    

	
@app.route("/sgPetStoreDogs.html")
def sgPetStoreDogs():
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
    cur.close()

@app.route("/PLCDogs.html")
def PLCDogs():
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
    cur.close()
    
@app.route("/PawsShopDogs.html")
def PawsShopDogs():
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
    cur.close()
    
@app.route("/sgPetStoreCats.html")
def sgPetStoreCats():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petStoreAnimal where petType ='cat' and petStoreId = 'SG Pet Store'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("sgPetStoreCats.html", cat=cat)
    cur.close()

@app.route("/PLCCats.html")
def PLCCats():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petStoreAnimal where petType ='cat' and petStoreId = 'Pet Lovers Center'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("PLCCats.html", cat=cat)
    cur.close()
    
@app.route("/PawsShopCats.html")
def PawsShopCats():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select * from petStoreAnimal where petType ='cat' and petStoreId = 'Paws Shop'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    #to include the values
    return render_template("PawsShopCats.html", cat=cat)
    cur.close()

@app.route("/SG Pet Store/buydog/<name>")
def sgPetStoreDogBreedBuy(name):
    cur = mysql.connection.cursor()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'SG Pet Store'  and name = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='SG Pet Store' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
    value = cur.execute(sql)
    others = cur.fetchall()
    return render_template("sgPetStoreDogBreedBuy.html", result=result, others=others,img=name)
    cur.close()
    
@app.route("/Pet Lovers Center/buydog/<name>")
def PLCDogBreedBuy(name):
    cur = mysql.connection.cursor()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Pet Lovers Center'  and name = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Pet Lovers Center' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
    value = cur.execute(sql)
    others = cur.fetchall()
    return render_template("PLCDogBreedBuy.html", result=result, others=others,img=name)
    cur.close()
    
@app.route("/Paws Shop/buydog/<name>")
def PawsShopDogBreedBuy(name):
    cur = mysql.connection.cursor()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Paws Shop'  and name = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Paws Shop' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
    value = cur.execute(sql)
    others = cur.fetchall()
    return render_template("PawsShopDogBreedBuy.html", result=result, others=others,img=name)
    cur.close()
    
@app.route("/SG Pet Store/buycat/<name>")
def sgPetStoreCatBreedBuy(name):
    cur = mysql.connection.cursor()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'SG Pet Store'  and name = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='SG Pet Store' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
    value = cur.execute(sql)
    others = cur.fetchall()
    return render_template("sgPetStoreCatBreedBuy.html", result=result,others=others, img=name)
    cur.close()
    
@app.route("/Pet Lovers Center/buycat/<name>")
def PLCCatBreedBuy(name):
    cur = mysql.connection.cursor()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Pet Lovers Center'  and name = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Pet Lovers Center' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
    value = cur.execute(sql)
    others = cur.fetchall()
    return render_template("PLCCatBreedBuy.html", result=result,others=others, img=name)
    cur.close()
    
@app.route("/Paws Shop/buycat/<name>")
def PawsShopCatBreedBuy(name):
    cur = mysql.connection.cursor()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petstoreanimal.petStoreId = 'Paws Shop'  and name = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    sql = "select IC, petStoreAnimal.petStoreID, petStoreAnimal.Name, PetType, DateOfBirth, Gender, Breed, Price, Size, HDB, petstore.telephone, petstore.email, petstore.address from petStoreAnimal join petstore on petstoreanimal.petStoreID = petstore.petStoreID where petStoreAnimal.petStoreID ='Paws Shop' and name not in  (select name from petStoreAnimal where name = '" + name + "') and breed in ( select breed from petStoreAnimal where name ='" + name + "')"
    value = cur.execute(sql)
    others = cur.fetchall()
    return render_template("PawsShopCatBreedBuy.html", result=result,others=others, img=name)
    cur.close()



if __name__ =="__main__":
   

	#app.debug = True
	app.run()
    