import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random as rn
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def prepare_dataset():
    record = []
    for c in classes:
        path = os.path.join(dir, c)
        lable = classes.index(c)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = np.array(image, dtype=np.float32)
            # print(image)
            record.append([image, lable])

    # print(record)
    img_pic = open('data.pickle', 'wb')
    pickle.dump(record, img_pic)
    img_pic.close()


def load_dataset():
    img_pic = open('data.pickle', 'rb')
    data = pickle.load(img_pic)
    img_pic.close()

    np.random.shuffle(data)
    feature = []
    lables = []

    for img, lable in data:
        feature.append(img)
        lables.append(lable)
    feature = np.array(feature, dtype=np.float32)
    lables = np.array(lables)

    feature = feature / 255.0
    return [feature, lables]

# MODEL
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='Same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))
# COMPILE
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Performing data augmentation to avoid model overfitting.

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

prepare_dataset()
(feature,labels) = load_dataset()
X_train, X_test, Y_train, Y_test = train_test_split(feature,labels,test_size = 0.1,random_state=42)
classes = ['daisy','dandelion','rose','sunflower','tulip']
datagen.fit(X_train)

batch_size = 100
epochs = 20

History = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,Y_test),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)

loss_score, accuracy_score = model.evaluate(X_test, Y_test, verbose=1)
print('Accuracy of model', accuracy_score)
predictions = model.predict(X_test)

obtained = []
truth = []
for i in range(len(predictions)) :
    obtained.append(classes[np.argmax(predictions[i])])
    truth.append(classes[Y_test[i]])
cm = confusion_matrix(truth,obtained)
print(classification_report(truth,obtained))
