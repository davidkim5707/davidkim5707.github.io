---
layout: post
title: Kaggle_Aerial Cactus Identification
categories:
  - Kaggle Competition
tags:
  - Tensorflow
  - KNN
  - Deep-Learing
last_modified_at: 2021-01-25
use_math: true
---

'''python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '/kaggle/input/aerial-cactus-identification'
labels = pd.read_csv(data_dir+"/train.csv")
labels['has_cactus'] = labels['has_cactus'].astype('str')

labels.head()
import zipfile
train_dir = "/kaggle/temp/train"
test_dir = "/kaggle/temp/test"
with zipfile.ZipFile(data_dir+"/train.zip","r") as z:
    z.extractall("/kaggle/temp")
with zipfile.ZipFile(data_dir+"/test.zip","r") as z:
    z.extractall("/kaggle/temp")
for dirname, _, filenames in os.walk('/kaggle/temp'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
csv_path = '../input/aerial-cactus-identification/train.csv'

df = pd.read_csv(csv_path)
df.head()
filename = df['id']
filename.head()
file_paths = [os.path.join(train_dir, fname) for fname in filenames]
file_paths[:10]
train_df = pd.DataFrame(data={'id': file_paths, 'has_cactus': df['has_cactus']})
train_df.head()
train_df = train_df.astype(np.str)
sample_csv_path = '/kaggle/input/aerial-cactus-identification/sample_submission.csv'
sample_df = pd.read_csv(sample_csv_path)
len(train_df)
train_df = train_df[:-500]
test_df = train_df[-500:]
path = train_df['id'][0]
#  Load Packages
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
os.listdir(test_dir)
 # Data Explore
path
img_pil = Image.open(path)
image = np.array(img_pil)

image.shape
plt.imshow(image)
plt.show()
# Hyperparameter
input_shape = (32,32,3)
batch_size = 32
num_classes =2
num_epochs = 10

learning_rate = 0.01
# Model
inputs = layers.Input(input_shape)
net = layers.Conv2D(64, (3, 3), padding='same')(inputs)
net = layers.Conv2D(64, (3, 3), padding='same')(net)
net = layers.Conv2D(64, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)

net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.Conv2D(128, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.Conv2D(256, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.Conv2D(512, (3, 3), padding='same')(net)
net = layers.BatchNormalization()(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D(pool_size=(2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=['accuracy'])
# Data Preprocess
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255.
)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='id',
    y_col='has_cactus',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='id',
    y_col='has_cactus',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='sparse'
)
# Train
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)
# Evaluate
preds = []

for fname in tqdm_notebook(sample_df['id']):
    path = os.path.join(test_dir, fname)

    img_pil = Image.open(path)
    image = np.array(img_pil)

    pred = model.predict(image[tf.newaxis, ...])
    pred = np.argmax(pred)
    preds.append(pred)

submission_df = pd.DataFrame(data={'id': sample_df['id'], 'has_cactus': preds})
submission_df.to_csv('samplesubmission.csv', index=False)
'''

