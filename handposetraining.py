# -*- coding: utf-8 -*-
"""HandPoseTraining.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16Hie6_ZH4buv6NdCNOvxN-XuezR5N3aO
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#Download Folder
from google.colab import files

#read in data using pandas
train_df = pd.read_csv('handData.csv')
#check data has been read in properly
print(train_df)

#create a dataframe with all training data except the target column
X = train_df.drop(columns=["0"])

#check that the target variable has been removed
print(X)

X.shape

data_y = train_df["0"]
data_y[0:5]
data_y.shape

def MultiLabelMaker(y, No_labels):
    train_Y = []
    for i in y:
        labels = np.array([0]*No_labels)
        labels[i-1] = 1
        train_Y.append(labels)
    return np.array(train_Y)

Y = MultiLabelMaker(data_y, 4)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#create model
model = Sequential()

#get number of columns in training data
n_cols = X.shape[1]

#add layers to model
model.add(Flatten(input_shape=(n_cols,)))
model.add(Dense(64, activation='relu' ))
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

def model(X):
    #create model
    model = Sequential()

    #get number of columns in training data
    n_cols = X.shape[1]

    #add layers to model
    model.add(Flatten(input_shape=(n_cols,)))
    model.add(Dense(64, activation='relu' ))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model

model_2 = model(X_train)

#compile model using accuracy to measure model performance
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
model_2.fit(X_train, y_train, epochs=30, validation_split=0.2)

prediction = model_2.predict(X_test)

model_2.save("handPose")

!zip -r /content/handPose.zip /content/handPose
files.download("/content/handPose.zip")