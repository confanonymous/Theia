import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
batch_size =128
nb_classes = 30
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

def cnn(drop_rate=0.1):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    return model

conv_model = cnn()
history = conv_model.fit(X_train, y_train, shuffle=True,epochs=10, verbose=1, 
                        batch_size=batch_size, validation_data=(X_test, y_test))