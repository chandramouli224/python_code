import keras
import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Input, Dropout
from keras.models import Model
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
from tensorflow.keras import layers
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

def build_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = MobileNet(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(224, 224, 3),
        pooling='avg')
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers

    op = Dense(224, activation='relu')(base_model.output)
    op = Dropout(.25)(op)

    ##
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    ##
    output_tensor = Dense(1, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

def train_model():
    # base_model = MobileNet(
    #     include_top=False,
    #     weights='imagenet',
    #     input_shape=(224, 224, 3)
    # )  # Do not include the ImageNet classifier at the top (1000 => 2)
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1280,activation='relu',kernel_initializer=tf.keras.initializers.glorot_uniform(42), bias_initializer='zeros')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    predictions = tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    model = tf.keras.models.Model(inputs=base_model.input,outputs = predictions)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    loss = "categorical_crossentropy"

    for layer in model.layers:
        layer.trainable = True

    model.summary()

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    initial_epochs = 10
    validation_steps = 5
    # history = model.fit(X_train,
    #                     epochs=initial_epochs,
    #                     validation_split=0.2,
    #                     # validation_data=X_test,
    #                     callbacks=[tensorboard_callback])

    # base_model.trainable = False
    # inputs = keras.Input(shape=(224, 224, 3))
    # data_augmentation = keras.Sequential(
    #     [
    #         layers.experimental.preprocessing.RandomFlip("horizontal"),
    #         layers.experimental.preprocessing.RandomRotation(0.1),
    #     ]
    # )
    # x = data_augmentation(inputs)  # Apply random data augmentation
    #
    # # Pre-trained Xception weights requires that input be normalized
    # # from (0, 255) to a range (-1., +1.), the normalization layer
    # # does the following, outputs = (inputs - mean) / sqrt(var)
    # norm_layer = keras.layers.experimental.preprocessing.Normalization()
    # mean = np.array([127.5] * 3)
    # var = mean ** 2
    # # Scale inputs to [-1, +1]
    # x = norm_layer(x)
    # norm_layer.set_weights([mean, var])
    #
    # # The base model contains batchnorm layers. We want to keep them in inference
    # # mode when we unfreeze the base model for fine-tuning, so we make sure that the
    # # base_model is running in inference mode here.
    # x = base_model(x, training=False)
    # x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    # outputs = keras.layers.Dense(1)(x)
    # model = keras.Model(inputs, outputs)

    return model