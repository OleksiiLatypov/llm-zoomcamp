import os
import shutil
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

#print(tf.__version__)

print(len(os.listdir('./data/train/straight')))
print(len(os.listdir('./data/train/curly')))

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def create_cnn_model(lr=0.002, momentum=0.8):
    inputs = keras.Input(shape=(200, 200,3))
    # Convolutional layer with 32 filters and ReLU activation
    x = layers.Conv2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu')(inputs)

     # MaxPooling layer to reduce feature map size
    x= layers.MaxPooling2D(pool_size=(2,2))(x)

    #Flattenization
    x = layers.Flatten()(x)

    #Dense layer
    x = layers.Dense(64, activation='relu')(x)

    #output
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    optimizer = optimizers.SGD(learning_rate=lr, momentum=0.8)
    loss = losses.BinaryCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

cnn_model = create_cnn_model()
cnn_model.summary()



train_dir = './data/train'
val_dir = './data/test'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200, 200), batch_size=20, shuffle=True, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(200, 200), batch_size=20, shuffle=True, class_mode='binary')

print(train_generator, val_datagen)


history = cnn_model.fit(train_generator, epochs=10, validation_data = val_generator)


train_accuracy = history.history['accuracy']
print(train_accuracy)
train_loss = history.history['loss']
print(train_loss)



train_datagen_augment = ImageDataGenerator(rescale=1./255,
                                          rotation_range=50,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          zoom_range=0.1,
                                          horizontal_flip=True,
                                          fill_mode='nearest')

train_generator_augment = train_datagen_augment.flow_from_directory(train_dir,
                                                                    target_size=(200, 200),
                                                                    batch_size=20,
                                                                    shuffle=True,
                                                                    class_mode='binary')


history_cnn_augment = cnn_model.fit(train_generator_augment, epochs=10, validation_data=val_generator)


test_loss = history_cnn_augment.history['val_loss']
np.mean(test_loss)


test_accuracy = history_cnn_augment.history['val_accuracy']
test_accuracy_for_last_five = test_accuracy[5:10]
np.mean(test_accuracy_for_last_five)



cnn_model.save('cnn_model.h5')