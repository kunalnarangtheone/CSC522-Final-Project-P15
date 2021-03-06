import numpy as np
import pandas as pd
import os
import gc

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
from keras.models import Model, load_model

import keras.backend as K
from keras.models import Sequential

from keras.utils import plot_model
import gc

# Kaggle and Shell commands
# !pip install kaggle
# !mkdir ~/.kaggle/
# !ech,\"key\":\"\"}" > ~/.kaggle/kaggle.json
# !chmod 600 ~/.kaggle/kaggle.json
# !cat ~/.kaggle/kaggle.json

# !kaggle competitions download -c humpback-whale-identification

# !mkdir input
# !mv train.csv input/train.csv
# !unzip /content/train.zip -d input/train >/dev/null
# !unzip /content/test.zip -d input/test >/dev/null

# Loading input data from https://www.kaggle.com/pestipeti/keras-cnn-starter/
train_df = pd.read_csv("input/train.csv")
train_df.head()

mv /content/sample_submission.csv /content/input/

train_df = pd.read_csv("input/train.csv")

def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("input/"+dataset+"/"+fig, target_size=(100, 100, 3))
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
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

X = prepareImages(train_df[:12000], train_df[:12000].shape[0], "train")
X /= 255

y, label_encoder = prepare_labels(train_df[:12000]['Id'])

model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), strides = (1,1)))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3)))
model.add(Flatten())
model.add(Dense(1000, activation="relu"))
model.add(Dropout(0.7))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()


plot_model(model, to_file='model.png')

history = model.fit(X, y, epochs=100, batch_size=16, verbose=1)
gc.collect()
model.save(model.h5)