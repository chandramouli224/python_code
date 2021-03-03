import numpy as np
import cv2
import mobilenet
from keras.applications import imagenet_utils
#this line is used to load pretrained yolo algorithm
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes = []

#import_mobile will load pretrained mobile net deeplearning algorithm from mobile net class which I have created
model = mobilenet.import_model()

#this is just to know look at on which classes yolo was trained
with open('./model_data/coco_classes.txt','r') as f:
  classes = [line.strip() for line in f.readlines()]
print(classes)

#reading all layers of yolo
layer_names = net.getLayerNames()
#reading output layers
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers() ]

#cars3.jpg
img = cv2.imread('cars3.jpg')
#resizing image fx, fy are scale factors which are used to scale image along x and y axis
img = cv2.resize(img, None, fx=0.4, fy=0.4)

height, width, channels = img.shape

#blobFromImage function is used to preprocess(mean subtraction, normalizing, and channel swapping) the images before sending it to NN
#https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416),(0,0,0), True, crop = False)
#print(blob)
# for b in blob:
#    for n,img_blob in enumerate(b):
#      cv2.imshow(str(n),img_blob)

net.setInput(blob)

#this out list of lists that contains the detected objects 1st four elements are center(x,y), height and width of bounding box of detected object
# rest of the elements are scores for each class
outs = net.forward(outputlayers)
#print(outs)

class_ids = []
confidences =[]
boxes =[]

for out in outs:
  for detection in out:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    if confidence >0.5:
      #calculating object center(x,y), w, h with respect to actual size of the image
      center_x = int(detection[0] * width)
      center_y = int(detection[1] * height)
      w = int(detection[2] * width)
      h = int(detection[3] * height)

      cv2.circle(img, (center_x, center_y),10,(255,0,0),2)
      x = int(center_x - w / 2)
      y = int(center_y - h / 2)
      #cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),2)

      boxes.append([x,y,w,h])
      confidences.append(float(confidence))
      class_ids.append(class_id)
#it reduces the number of boxes in to one box for an object. lets say 3 different boxes were detected for single objects then below function will reduce 3 boxes into one box if an overlap exists among the boxes.
indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
print(indexes)
#print(boxes)
print(len(indexes))
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
  if i in indexes:
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])

    #detected objects are cropped so that it we feed that particular object as input to mobilenet for object classification
    frame = img[y:y+h,x:x+w]
    #22,224 is one of the standard img size that mobilenet supports
    frame =  cv2.resize(frame,(224,224))
    preprocessed_image = mobilenet.prepare_image(frame)
    #predicting the class of the cropped object
    predictions = model.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    #uncomment below line to see cropped images
    #cv2.imshow('test',frame), cv2.waitKey(0), cv2.destroyAllWindows()
    cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),3)
    cv2.putText(img, str(max(results[0])[1]),(x,y+30),font, 1, (0,0,250))


cv2.putText(img, str(len(indexes))+" cars detected",(10,10),font, 1, (0,0,250))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##
### Create a video capture object
# cap = cv2.VideoCapture('example1.mp4')
# ##
# ### Check video properties
# print('FPS: \t\t'+str(cap.get(cv2.CAP_PROP_FPS)))
# print('No. of Frames: \t'+str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
# print('Frame width: \t'+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
# print('Frame height: \t'+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# ##
# for i in range(0,10): # reads first 10 frames one by one (comment this line and uncomment next line to run on full video)
#
#     # Capture a frame
#   ret, frame = cap.read()
#
#     # Perform any operation on the frame here
#   edges = cv2.Canny(frame,100,200) # image, min/max gradient thresholds
#   blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416),(0,0,0), False, crop = False)
#     # Drawing shapes on the image
#     #rect = cv2.rectangle(frame,(200,200),(400,400),(0,0,255),2) # image, top-left corner, bottom-right corner, color (BRG), line width
# ##    # Writing text on the image
#     #cv2.putText(rect, 'Lena', (260,440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) # image, text, location, fontFace, fontScale, color
#     #cv2.imshow('class',rect)
#   net.setInput(blob)
#   outs = net.forward(outputlayers)
#   test(outs,frame)
#     #print(frame.shape)
# ##    #cv2_imshow(edges)
# ##    #cv2.waitKey(0), cv2.destroyAllWindows()
# ##    # Display the resulting frame (or save it as an image, or skip this step)
# ##    #cv2_imshow(frame) #cv2.imshow('frame',frame)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
# ##
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
