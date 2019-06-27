from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
# from keras.utils.vis_utils import plot_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
nb_classes = 10
nb_epoch = 200

img_rows, img_cols = 32, 32
img_channels = 3

print('Loading data...')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('X_train shape:', X_train.shape)
print('Y_test shape:', Y_test.shape)

model = Sequential()

# Note: it expects the data_format='channels_first'
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(img_channels, img_rows, img_cols)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# plot_model(model, to_file='model.png')

X_train = X_train.astype('float32') - 128
X_test = X_test.astype('float32') - 128
X_train /= 128
X_test /= 128

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True,
          callbacks=[ModelCheckpoint('weights/cifar.h5', save_best_only=False)])