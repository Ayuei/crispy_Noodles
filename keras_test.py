from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras

def main(x,y):

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(128,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy']
                  ,optimizer=keras.optimizers.SGD())
    model.fit(x, y, verbose=1, batch_size=1, epochs=10)

    score = model.evaluate(x, y)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
