from flask import Flask,Blueprint, render_template, Flask, redirect, url_for, render_template, request, g
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

def getCatPrediction(filename):
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import decode_predictions

    model3 = VGG16()
    image = load_img('static/temp.jpg', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model3.predict(image)
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

def getCatPrediction(filename):
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import decode_predictions

    model3 = VGG16()
    image = load_img('static/temp.jpg', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model3.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    return label[1], label[2]*100


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

#CAT IMAGE CAPTION CODE STARTS HERE

 
# extract features from each photo in the directory
def extract_features(filename):
    
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.preprocessing.image import load_img
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model
    import time
    
	# load the model
    model4 = VGG16()
	# re-structure the model
    model4 = Model(inputs=model4.inputs, outputs=model4.layers[-2].output)
	# load the photo
    image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
    image = img_to_array(image)
	# reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
    image = preprocess_input(image)
	# get features
    feature = model4.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

        
   # if(filetype.lower() != "png" or filetype.lower() != "jpg" ):
        

    

@catClassifyClass.route("/catclassify", methods= ['POST'])
def catclassify():
    if request.method == "POST":
        import io
        from tensorflow.keras import backend as K
        from pickle import load
        import os
        import filetype
        K.clear_session()
    #load the model
        model = load_model("static/CatIdentification/catmodel.hdf5")
    #get the image of the cat for prediction
        img = request.files['imagefile']
        
        #Check if 
        checkT = fileValidationCheck(img)
       
        if ( checkT == "checkT"):
            checkT = True
            checkTType = False
            return render_template("/catclassify.html", checkT=checkT, checkTType=checkTType)
 
        if ( checkT == "checkTType"):
            checkT = False
            checkTType = True
            return render_template("/catclassify.html", checkT=checkT, checkTType=checkTType)
  
        
  
        checkT = False
        checkTType = False
        
        img1 = request.files["imagefile"].read()
        img_path = "static/temp.jpg"
        pic = Image.open(img)
        pic.save(img_path, 'JPEG')
        
        x = path_to_tensor(img_path)
        tensors = x.astype('float32')
        preprocessed_input = preprocess_input_vgg19(tensors)
        y = VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

        predictions = model.predict([y])
        breed_predictions = [np.argmax(prediction) for prediction in predictions]
        #catName = cat_names[breed_predictions[0]]
        catName = ntpath.basename(cat_names[breed_predictions[0]])
        
        def get_predection(image,net,LABELS,COLORS):
            import time
            from werkzeug.utils import secure_filename
            (H, W) = image.shape[:2]
            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            
            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)

            end = time.time()
        
            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]

                    classID = np.argmax(scores)
                    confidence = scores[15]
                    
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > confthres:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        
                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        
                        # apply non-maxima suppression to suppress weak, overlapping bounding
                        # boxes
                        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                                                nmsthres)
                        
                        # ensure at least one detection exists
                        if len(idxs) > 0:
                            # loop over the indexes we are keeping
                            for i in idxs.flatten():
                                # extract the bounding box coordinates
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])
                                
                                # draw a bounding box rectangle and label on the image
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                y = y - 10 if y - 10 > 10 else y + 15
                                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                                
                                img = request.files["imagefile"]
                                filename = secure_filename(img.filename)
                                getCatPrediction(filename)
                                labelx, acc = getCatPrediction(filename)

                                cv2.putText(image, catName, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                                return image
                            
        #get the image of the dog for prediction
        
        # load our input image and grab its spatial dimensions
        yolo = True

        try:
            #img1 = request.files["imagefile"].read()
            img1 = Image.open(io.BytesIO(img1))
            npimg=np.array(img1)
            image=npimg.copy()
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            res=get_predection(image,nets,Lables,Colors)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

            cv2.waitKey()
            cv2.imwrite("filename1.png", res)
            np_img=Image.fromarray(image)
            img_encoded=image_to_byte_array(np_img)  
            base64_bytes = base64.b64encode(img_encoded).decode("utf-8")     
        except:
            yolo = False
        
        var = gTTS(catName, lang = 'en')
        var.save("static/catsound.mp3")
    
    
    # generate a description for an image
# generate a description for an image
    def generate_desc(model, tokenizer, photo, max_length):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from pickle import load
        from numpy import argmax
	# seed the generation process
        in_text = 'a'
	# iterate over the whole length of the sequence
        for i in range(max_length):
		# integer encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
            yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
            yhat = argmax(yhat)
		# map integer to word
            word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
            if word is None:
                break
		# append as input for generating the next word
            in_text += ' ' + word
		# stop if we predict the end of the sequence
            if word == 'cat':
                break
        return in_text
    
    # load the tokenizer
    tokenizer = load(open('static/CatIdentification/CatImageCaption/tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 8
    # load the model
    model = load_model('static/CatIdentification/CatImageCaption/model-ep007-loss0.056-val_loss0.225.h5')
    # load and prepare the photograph
    photo = extract_features(img_path)

    #get description from database
    cur = mysql.connection.cursor()
    sql = "select Breed, Description, AverageLifeSpan from cat where naming_patter = '" + catName + "'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    cur.close()

    # generate description
    description1 = generate_desc(model, tokenizer, photo, max_length)
    description = cat[0][0] + ' is ' + description1
    if yolo:
        return render_template("catclassify.html", img_path = base64_bytes, description = description, cat = cat)
    else:
        return render_template("catclassify.html", noyolo = img_path, description = description, cat = cat)
    
       
       # generate a description for an image
    # generate a description for an image
    def generate_desc(model, tokenizer, photo, max_length):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from pickle import load
        from numpy import argmax
    	# seed the generation process
        in_text = 'a'
    	# iterate over the whole length of the sequence
        for i in range(max_length):
    		# integer encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
    		# pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
    		# predict next word
            yhat = model.predict([photo,sequence], verbose=0)
    		# convert probability to integer
            yhat = argmax(yhat)
    		# map integer to word
            word = word_for_id(yhat, tokenizer)
    		# stop if we cannot map the word
            if word is None:
                break
    		# append as input for generating the next word
            in_text += ' ' + word
    		# stop if we predict the end of the sequence
            if word == 'cat':
                break
            return in_text
    
    # load the tokenizer
    tokenizer = load(open('static/CatIdentification/CatImageCaption/tokenizer.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    max_length = 8
    # load the model
    model = load_model('static/CatIdentification/CatImageCaption/model-ep007-loss0.056-val_loss0.225.h5')
    # load and prepare the photograph
    photo = extract_features(img_path)

    #get description from database
    cur = mysql.connection.cursor()
    sql = "select Breed, Description, AverageLifeSpan from cat where naming_patter = '" + catName + "'"
    value = cur.execute(sql)
    cat = cur.fetchall()
    cur.close()

    # generate description
    description1 = generate_desc(model, tokenizer, photo, max_length)
    description = cat[0][0] + ' is ' + description1
    if yolo:
        return render_template("catclassify.html", img_path = base64_bytes, description = description, cat = cat)
        checkT = False
        checkTType = False
    else:
        return render_template("catclassify.html", noyolo = img_path, description = description, cat = cat)
        checkT = False
        checkTType = False
    
       
      
     
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
    