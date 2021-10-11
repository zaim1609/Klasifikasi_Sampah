import os

fold_cardboard = os.path.join('/content/drive/MyDrive/data/Garbage classification/Garbage classification/cardboard')
fold_glass = os.path.join('/content/drive/MyDrive/data/Garbage classification/Garbage classification/glass')
fold_metal = os.path.join('/content/drive/MyDrive/data/Garbage classification/Garbage classification/metal')
fold_paper = os.path.join('/content/drive/MyDrive/data/Garbage classification/Garbage classification/paper')
fold_plastic = os.path.join('/content/drive/MyDrive/data/Garbage classification/Garbage classification/plastic')


print('jumlah kardus = ', len(os.listdir(fold_cardboard)))
print('jumlah kaca = ', len(os.listdir(fold_glass)))
print('jumlah besi = ', len(os.listdir(fold_metal)))
print('jumlah kertas = ', len(os.listdir(fold_paper)))
print('jumlah plastik = ', len(os.listdir(fold_plastic)))

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

BASIS_DIR = '/content/drive/MyDrive/data/Garbage classification/Garbage classification'
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    #augmentasi
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    #memecah data menjadi data training dan data validasi
    validation_split=0.2
)

#pelebelan data menggunakan image data generator bedasarkan folder

train_generator = training_datagen.flow_from_directory(
    BASIS_DIR,
    target_size = (300,300),
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
    subset='training'
)

validation_generator = training_datagen.flow_from_directory(
    BASIS_DIR,
    target_size = (300,300),
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
    subset='validation'
)

from tensorflow.keras.callbacks import ModelCheckpoint

#membuat model jarigan saraf tiruan
#model yg digunakan = sequential

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300,300, 3)), 
  tf.keras.layers.MaxPooling2D(2, 2),
 
    
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
 
    
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
    
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  
    
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(5, activation='softmax')
])



model.summary()


#melakukan kompilasi model

model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history=model.fit(
    train_generator,
    validation_steps = 14,
    epochs=100,
    steps_per_epoch=59,
    validation_data=validation_generator
    
)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label = 'akuarasi Training')
plt.plot(epochs, val_acc,'b',label='akurasi validasi')
plt.title('akurasi training dan validasi')
plt.legend(loc=0)
plt.figure()
plt.show()

plt.plot(epochs, loss, 'r', label = 'loss training')
plt.plot(epochs, val_loss,'b',label='loss validasi')
plt.title('loss training dan validasi')
plt.legend(loc=0)
plt.figure()
plt.show()
