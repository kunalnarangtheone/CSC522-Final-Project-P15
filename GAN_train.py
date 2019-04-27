import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

from gan import CGAN

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

train_df = pd.read_csv("./input/train.csv")
train_df.head()

# Loading input data from https://www.kaggle.com/pestipeti/keras-cnn-starter/
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("./input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    y = integer_encoded

    # onehot_encoder = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # # print(onehot_encoded)

    # y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

X_train = prepareImages(train_df[:3000], 3000, "train")
X_train /= 255

y_train, label_encoder = prepare_labels(train_df[:3000]['Id'])

if __name__ == '__main__':
    cgan = CGAN(100, 100, 3, len(label_encoder.classes_))
    cgan.train(epochs=1000001, start = 0, X_train = X_train, y_train = y_train, batch_size=32, sample_interval=200)