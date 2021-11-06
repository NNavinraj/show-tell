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
dogClassifyClass = Blueprint("dogClassifyClass", __name__, static_folder="static", template_folder="templates")
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

@dogClassifyClass.route("/dogclassify.html", methods= ['POST'])
def dogclassify():

    if request.method == "POST":
        import io
        from tensorflow.keras import backend as K
        
        K.clear_session()
    #read the csv file

        df_labels = pd.read_csv("static/DogIdentification/labels.csv")
    #store training and testing images folder location
        train_file = "static/DogIdentification/Train"
        test_file = "static/DogIdentification/Test"

    #specify number
        num_breeds = 55
        im_size = 224
        batch_size = 64
        encoder = LabelEncoder()

    #get only 55 unique breeds record 
        breed_dict = list(df_labels['breed'].value_counts().keys()) 
        new_list = sorted(breed_dict,reverse=True)[:num_breeds]
    #change the dataset to have only those 60 unique breed records
        df_labels = df_labels.query('breed in @new_list')

    #load the model
        model2 = load_model("static/DogIdentification/dogmodel")
        
        img2 = request.files['imagefile']
        img = request.files["imagefile"].read()
        img_path = "static/temp.jpg"
        #img_path = img_path + img2.filename
        pred_img_path = img_path
        pic = Image.open(img2)
        pic.save(img_path, 'JPEG')
    #read the image file and convert into numeric format
    #resize all images to one dimension i.e. 224x224
        pred_img_array = cv2.resize(cv2.imread(pred_img_path,cv2.IMREAD_COLOR),((im_size,im_size)))
    #scale array into the range of -1 to 1.
    #expand the dimension on the axis 0 and normalize the array values
        pred_img_array = preprocess_input(np.expand_dims(np.array(pred_img_array[...,::-1].astype(np.float32)).copy(), axis=0))
     
    #feed the model with the image array for prediction
        pred_val = model2.predict(np.array(pred_img_array,dtype="float32"))
     
    #display the predicted breed of dog
        pred_breed = sorted(new_list)[np.argmax(pred_val)]
        
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
                    confidence = scores[classID]
                    
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
                                getPrediction(filename)
                                labelx, acc = getPrediction(filename)

                                cv2.putText(image, pred_breed, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                                return image
        

        
    #get the image of the dog for prediction
        
        # load our input image and grab its spatial dimensions
        try:
            #img = request.files["imagefile"].read()
            img = Image.open(io.BytesIO(img))
            npimg=np.array(img)
            image=npimg.copy()
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            res=get_predection(image,nets,Lables,Colors)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

            cv2.waitKey()
            cv2.imwrite("filename.png", res)
            np_img=Image.fromarray(image)
            img_encoded=image_to_byte_array(np_img)  
            base64_bytes = base64.b64encode(img_encoded).decode("utf-8")  

            #get description from database
            cur = mysql.connection.cursor()
            sql = "select Breed, Description, AverageLifeSpan from dog where naming_patter = '" + pred_breed + "'"
            value = cur.execute(sql)
            description = cur.fetchall()
            cur.close()
            var = gTTS(str(description[0][0]), lang = 'en')
            var.save("static/dogsound.mp3")
            return render_template("dogclassify.html", img_path = base64_bytes, description = description)
        except:

    #get description from database
            cur = mysql.connection.cursor()
            sql = "select Breed, Description, AverageLifeSpan from dog where naming_patter = '" + pred_breed + "'"
            value = cur.execute(sql)
            description = cur.fetchall()
            cur.close()
            var = gTTS(str(description[0][0]), lang = 'en')
            var.save("static/dogsound.mp3")
            return render_template("dogclassify.html", noyolo = pred_img_path, description = description)

@dogClassifyClass.route("/dogclassify", methods= ['GET'])
def viewDog():
    if request.method == "GET":
        return render_template("/dogclassify.html")
 


    