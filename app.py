from flask import Flask, redirect, url_for, render_template, request
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

@app.route("/dogclassify.html", methods= ['GET', 'POST'])
def dogclassify():
    if request.method == "GET":
        return render_template("dogclassify.html")
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

#extract VGG19 features
def extract_VGG19(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_vgg19(tensors)
    return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)
train_vgg19 = "static/CatIdentification/train_vgg19.pkl"
valid_vgg19 = "static/CatIdentification/valid_vgg19.pkl"
test_vgg19 = "static/CatIdentification/test_vgg19.pkl"


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
 
@app.route("/catclassify.html", methods= ['GET', 'POST'])
def catclassify():
    if request.method == "GET":
        return render_template("catclassify.html")
    if request.method == "POST":
        import io
        from tensorflow.keras import backend as K
        from pickle import load
        K.clear_session()
    #load the model
        model = load_model("static/CatIdentification/catmodel.hdf5")
    #get the image of the cat for prediction
        img = request.files['imagefile']
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

@app.route("/dogs.html")
def dogs():
    #insert sql statement to get names of dogs (seperated by breed size/HDB approved)
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
    cur.close()

@app.route("/cats.html")
def cats():
    #insert sql statement to get names of cats
    cur = mysql.connection.cursor()
    sql = "select Breed from cat"
    value = cur.execute(sql)
    cat = cur.fetchall()
    
    #to include the values
    return render_template("cats.html", cat=cat)
    cur.close()
    
@app.route("/dog/<name>")
def dogbreed(name):
    cur = mysql.connection.cursor()
    sql = "select Breed, AverageLifeSpan, Size, Description, Characteristic from dog where Breed = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    return render_template("dogbreed.html", result=result, img=name)
    cur.close()

@app.route("/cat/<name>")
def catbreed(name):
    cur = mysql.connection.cursor()
    sql = "select Breed, AverageLifeSpan, Size, Description, Characteristic from cat where Breed = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    return render_template("catbreed.html", result=result, img=name)
    cur.close()

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

@app.route("/buyheredog/dog/<name>")
def buydog(name):
    cur = mysql.connection.cursor()
    sql = " select IC, petstoreanimal.petStoreID,name,DateOfBirth,gender,vaccindated,breed,price,size,hdb, address, telephone, email from petstoreanimal join petstore on petstoreanimal.petstoreID = petstore.petStoreID where Breed = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    return render_template("buyheredog.html", result=result, img=name)
    cur.close()
    
@app.route("/buyherecat/cat/<name>")
def buycat(name):
    cur = mysql.connection.cursor()
    sql = " select IC, petstoreanimal.petStoreID,name,DateOfBirth,gender,vaccindated,breed,price,size,hdb, address, telephone, email from petstoreanimal join petstore on petstoreanimal.petstoreID = petstore.petStoreID where Breed = '" + name + "'"
    value = cur.execute(sql)
    result = cur.fetchall()
    return render_template("buyherecat.html", result=result, img=name)
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
	app.run(debug= True)