import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
def import_model():
    mobile = keras.applications.mobilenet.MobileNet()
    return mobile
def prepare_image(frame):
    img_path = ''
    #frame = image.load_img(frame , target_size=(224, 224))
    #print(frame.size)
    frame=frame.reshape(1,224, 224,3)
    # img_array = image.img_to_array(frame)
    # img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    # return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    return frame
# preprocessed_image = prepare_image('D:\Lectures\casestudy\dataset\German_Shepherd.jpeg')
# predictions = mobile.predict(preprocessed_image)
# results = imagenet_utils.decode_predictions(predictions)
# print(results)