from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.keras.utils import to_categorical

batch_size = 512
epoch = 30
img_rows, img_col = 28,28


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
cnn_model = Sequential()

cnn_model.add(Conv2D(64,(3, 3), input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())
cnn_model.add(Dense(32, activation = 'relu'))
cnn_model.add(Dense(10, activation = 'sigmoid'))

cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer='Adam',metrics =['accuracy'])

history = cnn_model.fit(x_train,
                        y_train,
                        batch_size = batch_size,
                        epochs = epoch,
                        verbose = 1,
                        validation_data = (x_test, y_test))