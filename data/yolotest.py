import numpy as np
import cv2
import mobilenet
import os
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import keras
from keras.applications import imagenet_utils
#this line is used to load pretrained yolo algorithm
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
# net = cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg')
classes = []


model = mobilenet.train_model()

with open('coco_classes.txt','r') as f:
  classes = [line.strip() for line in f.readlines()]


#reading all layers of yolo
layer_names = net.getLayerNames()
#reading output layers
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers() ]

def load_images_from_folder(folder,name):
    images = []
    x = []
    y = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,filename))
        # print(image)
        if image is not None:
          image=cv2.resize(image,(224,224))
          # img =img.reshape(1,224,224,3)
          x.append(image)
          y.append(name)
    return x,y


train_suv_x, train_suv_y = load_images_from_folder('./SUV/','SUV')
train_sedan_x,train_Sedan_y = load_images_from_folder('./Sedan/','SEDAN')
train_x=[]
train_y=[]
for i in range(0,len(train_sedan_x)):
  train_x.append(train_suv_x[i])
  train_y.append(0)
  train_x.append(train_sedan_x[i])
  train_y.append(1)
train_y = np.array(train_y).reshape(-1, 1)

# model.fit(np.array(train_x),np.array(train_y),batch_size=10, epochs=50,steps_per_epoch=10,verbose=1,validation_split=0.2)
# model.fit(np.array(train_x),np.array(train_y),batch_size=100, epochs=50,steps_per_epoch=20,validation_split=0.2)
# model.save('model.h5')
model = keras.models.load_model('model.h5')
pred = model.predict(np.array(train_x))
for i in range(len(pred)):
    if pred[i]>0.5:
        pred[i]=1
    else:
        pred[i]=0
print(confusion_matrix(train_y,pred))
#
# class_ids = []
# confidences =[]
# boxes =[]

## Create a video capture object
cap = cv2.VideoCapture('assignment-clip.mp4')
##
### Check video properties
print('FPS: \t\t'+str(cap.get(cv2.CAP_PROP_FPS)))
print('No. of Frames: \t'+str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('Frame width: \t'+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height: \t'+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
##
for j in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    # reads first 10 frames one by one (comment this line and uncomment next line to run on full video)
    class_ids = []
    confidences = []
    boxes = []
    # Capture a frame
    ret, frame = cap.read()
    height, width, channels = frame.shape
    img = frame
    # Perform any operation on the frame here
    edges = cv2.Canny(frame,100,200) # image, min/max gradient thresholds
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416),(0,0,0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # calculating object center(x,y), w, h with respect to actual size of the image
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width +20)
                h = int(detection[3] * height +20)

                # cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), 2)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        # it reduces the number of boxes in to one box for an object. lets say 3 different boxes were detected for single objects then below function will reduce 3 boxes into one box if an overlap exists among the boxes.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    suv=0
    sedan=0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # if label=='car':
            if label == 'car':
                # detected objects are cropped so that it we feed that particular object as input to mobilenet for object classification
                frame = img[y:y + h+10, x:x + w+10]
                if frame.size !=0:
                    # 224,224 is one of the standard img size that mobilenet supports
                    # print(frame.size)
                    frame = cv2.resize(frame, (224, 224))
                    preprocessed_image = mobilenet.prepare_image(frame)
                    # predicting the class of the cropped object
                    predictions = model.predict(preprocessed_image)
                    #print(predictions)
                    if predictions > 0.5:
                        results = 'SEDAN'
                        sedan=sedan+1
                    else:
                        results = "SUV"
                        suv=suv+1
                    # uncomment below line to see cropped images
                    # if label == 'car':
                    cv2.rectangle(img,(x, y), (x + w, y + h), (0, 255, 0), 3)
                    # cv2.putText(img, label,(x,y+30),font, 1, (0,0,250))
                    cv2.putText(img, results, (x, y + 30), font, 1, (0, 0, 250))
    cv2.putText(img, str(sedan +suv) + " cars detected", (10, 10), font, 1, (0, 0, 250))


        # cv2.putText(img, str(len(indexes)) + " cars detected", (10, 10), font, 1, (0, 0, 250))

    cv2.imshow('test', img)
    print("frame: "+str(j)+" sedan: "+str(sedan)+" suv: "+str(suv) + " total: "+str(sedan+suv))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


