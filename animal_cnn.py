from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.engine.sequential import relax_input_shape

classes = ["monkey","boar","bear","cat","giraffe","gorilla","horse","lion","penguin","rabbit","crow"]
num_classes = len(classes)
image_size = 150

X_train, X_test, y_train, y_test = np.load('animal.npy',allow_pickle=True)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32,(3,3), padding='same', input_shape =(150,150,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Activation('softmax'))

opt = optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=32, epochs=100)

score = model.evaluate(X_test,y_test,batch_size=32)