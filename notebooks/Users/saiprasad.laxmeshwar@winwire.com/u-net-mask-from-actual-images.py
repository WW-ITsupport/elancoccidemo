# Databricks notebook source
# MAGIC %md
# MAGIC # AIM
# MAGIC <div style = "text-align: justify">Using U-net, localize the area which contains tumor growth <b>(which cannot be easily determined by looking at the actual medical images)</b> and compare it against the mask images. Then by looking at the generated mask image, classify whether the tumor growth is <b>malignant, benign or normal.</b> Then we must also classify the masks.</div>
# MAGIC 
# MAGIC # Note
# MAGIC <div style = "text-align: justify">Later in the notebook, I have mentioned images taken from medical imaging as <b>real image</b> !!!</div>
# MAGIC 
# MAGIC # Dataset [Link](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)
# MAGIC 
# MAGIC # Please checkout the [U-net paper](https://arxiv.org/pdf/1505.04597.pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC # Import images

# COMMAND ----------

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras

# COMMAND ----------

import os

# COMMAND ----------

path = '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'

# COMMAND ----------

from keras.preprocessing.image import img_to_array

# COMMAND ----------

# MAGIC %md
# MAGIC **Helper function** to get the index for real image and mask.

# COMMAND ----------

def num (image) :
    
    val = 0
    
    for i in range(len(image)) :
        if image[i] == '(' :
            while True :
                i += 1
                if image[i] == ')' :
                    break
                val = (val*10) + int(image[i])
            break
    
    return val

# COMMAND ----------

# MAGIC %md
# MAGIC <div style = "text-align: justify">Initialize the arrays for benign, normal and malignant tumors, both real and mask. As already given the number of samples in benign, normal & malignant are <b>437, 133 and 210</b> respectively.</div>

# COMMAND ----------

X_b, y_b = np.zeros((437, 128, 128, 1)), np.zeros((437, 128, 128, 1))
X_n, y_n = np.zeros((133, 128, 128, 1)), np.zeros((133, 128, 128, 1))
X_m, y_m = np.zeros((210, 128, 128, 1)), np.zeros((210, 128, 128, 1))

# COMMAND ----------

for i, tumor_type in enumerate(os.listdir(path)) :
    for image in os.listdir(path+tumor_type+'/') :
        p = os.path.join(path+tumor_type, image)
        img = cv2.imread(p,cv2.IMREAD_GRAYSCALE)           # read image as  grayscale
        
        if image[-5] == ')' :
            
            img = cv2.resize(img,(128,128))
            pil_img = Image.fromarray (img)
            
            if image[0] == 'b' :
                X_b[num(image)-1]+= img_to_array(pil_img)  # If image is real add it
            if image[0] == 'n' :                           # to X as benign , normal
                X_n[num(image)-1]+= img_to_array(pil_img)  # or malignant.
            if image[0] == 'm' :
                X_m[num(image)-1]+= img_to_array(pil_img)
        else :
            img = cv2.resize(img,(128,128))
            pil_img = Image.fromarray (img)
            
            if image[0] == 'b' :
                y_b[num(image)-1]+= img_to_array(pil_img)  # Similarly add the target
            if image[0] == 'n' :                           # mask to y.
                y_n[num(image)-1]+= img_to_array(pil_img)
            if image[0] == 'm' :
                y_m[num(image)-1]+= img_to_array(pil_img)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize the results to verify the above method

# COMMAND ----------

plt.figure(figsize = (20,10))

for i in range(5) :
    plt.subplot(2,5,i+1)
    plt.imshow(X_b[i+1], 'gray')
    plt.title('Real Image')
    plt.axis('off')

for i in range(5) :
    plt.subplot(2,5,i+6)
    plt.imshow(y_b[i+1], 'gray')
    plt.title('Mask Image')
    plt.axis('off')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why did I take these pixelated masks instead of taking original size ?
# MAGIC <div style = "text-align: justify">I did try to take large image sizes, <b>but due to GPU and RAM constraints</b>, my kernel kept on crashing. So I went with smaller sizes. I encourage the reader to try some different sizes where masks are more accurate.</div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Create datasets for model training and validation

# COMMAND ----------

X = np.concatenate((X_b, X_n, X_m), axis = 0)
y = np.concatenate((y_b, y_n, y_m), axis = 0)

# COMMAND ----------

X /= 255.0
y /= 255.0

# COMMAND ----------

print(X.shape)
print(y.shape)

# COMMAND ----------

print(X.max())
print(X.min())

# COMMAND ----------

print(y.max())
print(y.min())

# COMMAND ----------

y[y > 1.0] = 1.0

# COMMAND ----------

print(y.max())
print(y.min())

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualization

# COMMAND ----------

plt.figure(figsize = (10,30))
i = 0
while i < 16 :
    
    x = np.random.randint(0,780)
    
    plt.subplot(8,2,i+1)
    plt.imshow(X[x],'gray')
    plt.title('Real Image')
    plt.axis('off')
    
    plt.subplot(8,2,i+2)
    plt.imshow(y[x],'gray')
    plt.title('Mask Image')
    plt.axis('off')
    
    i += 2
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC <div style = "text-align: justify"> <b>Take a good look at image 2 and 8</b> and think if the masks were not provided, then would it have been easy to know the location tumor. NO !!! This is the aim of U-net model, localize the abnormalities in the image itself. Let's see the implementation.</div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Train test split

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 1)

# COMMAND ----------

print(X_train.shape)
print(y_train.shape)

# COMMAND ----------

print(X_test.shape)
print(y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Creation [U-net](https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5)
# MAGIC <img src = "https://miro.medium.com/max/3600/1*f7YOaE4TWubwaFF7Z1fzNw.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Architecture
# MAGIC <div style = "text-align: justify">U-net architecture can localize the area of interest. It was first used in Biomedical imaging. The reason it is able to <b>distinguish and localize</b> the area is by classifying every pixel in the input image. <b>So the size of input and output images is the same</b>. It comprises of two paths - <b>Contracting path and Expanding path</b>.</div>
# MAGIC 
# MAGIC ### Contract Path
# MAGIC The Contracting path has two Convolutional layers and a Maxpooling layer.
# MAGIC 
# MAGIC ### Expansive Path
# MAGIC <div style = "text-align: justify">The Expanding path consists of both transpose Convolutional layer and two Convolutional layers. The corresponding image from contracting path is fed to this layer for precise predictions.</div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modifications
# MAGIC <div style = "text-align: justify">I kept the padding same so that I can get the mask of exact same dimensions as the actual image. The adam gradient descent was used with a small <b>learning rate of 0.00005</b>. Also I am planning to add BatchNormalization which was discovered after U-net. </div>

# COMMAND ----------

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose

from keras import Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Contracting path

# COMMAND ----------

inply = Input((128, 128, 1,))

conv1 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(inply)
conv1 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(conv1)
pool1 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv1)
drop1 = Dropout(0.2)(pool1)

conv2 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(drop1)
conv2 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(conv2)
pool2 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv2)
drop2 = Dropout(0.2)(pool2)

conv3 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(drop2)
conv3 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(conv3)
pool3 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv3)
drop3 = Dropout(0.2)(pool3)

conv4 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(drop3)
conv4 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(conv4)
pool4 = MaxPooling2D((2,2), strides = 2, padding = 'same')(conv4)
drop4 = Dropout(0.2)(pool4)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bottleneck layer

# COMMAND ----------

convm = Conv2D(2**10, (3,3), activation = 'relu', padding = 'same')(drop4)
convm = Conv2D(2**10, (3,3), activation = 'relu', padding = 'same')(convm)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expanding layer

# COMMAND ----------

tran5 = Conv2DTranspose(2**9, (2,2), strides = 2, padding = 'valid', activation = 'relu')(convm)
conc5 = Concatenate()([tran5, conv4])
conv5 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(conc5)
conv5 = Conv2D(2**9, (3,3), activation = 'relu', padding = 'same')(conv5)
drop5 = Dropout(0.1)(conv5)

tran6 = Conv2DTranspose(2**8, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop5)
conc6 = Concatenate()([tran6, conv3])
conv6 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(conc6)
conv6 = Conv2D(2**8, (3,3), activation = 'relu', padding = 'same')(conv6)
drop6 = Dropout(0.1)(conv6)

tran7 = Conv2DTranspose(2**7, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop6)
conc7 = Concatenate()([tran7, conv2])
conv7 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(conc7)
conv7 = Conv2D(2**7, (3,3), activation = 'relu', padding = 'same')(conv7)
drop7 = Dropout(0.1)(conv7)

tran8 = Conv2DTranspose(2**6, (2,2), strides = 2, padding = 'valid', activation = 'relu')(drop7)
conc8 = Concatenate()([tran8, conv1])
conv8 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(conc8)
conv8 = Conv2D(2**6, (3,3), activation = 'relu', padding = 'same')(conv8)
drop8 = Dropout(0.1)(conv8)

# COMMAND ----------

outly = Conv2D(2**0, (1,1), activation = 'relu', padding = 'same')(drop8)
model = Model(inputs = inply, outputs = outly, name = 'U-net')

# COMMAND ----------

keras.utils.plot_model(model, './model_plot.png', show_shapes = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Loss function
# MAGIC 
# MAGIC <div style = "text-align: justify">The loss for evaluating the performance of model in semantic segmentation will be <b>IoU (Intersection over Union)</b>. It is the ratio of intersection of pixels between predicted and target image over their union. The MeanIoU() method in tf.keras.metrics package can be used.</div>

# COMMAND ----------

from keras.metrics import MeanIoU

# COMMAND ----------

# MAGIC %md
# MAGIC # Training

# COMMAND ----------

model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.00005))
print(model.summary())

# COMMAND ----------

from keras.callbacks import ModelCheckpoint

# COMMAND ----------

checkp = ModelCheckpoint('./cancer_image_model.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)

# COMMAND ----------

history = model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_data = (X_test, y_test), callbacks = [checkp])

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Performance

# COMMAND ----------

plt.figure(figsize = (20,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'validation loss'])
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('Losses vs Epochs', fontsize = 15)

# COMMAND ----------

from keras.models import load_model
model = load_model('./cancer_image_model.h5')

# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

print(y_pred.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictions

# COMMAND ----------

plt.figure(figsize = (20,80))

i = 0
x = 0
while i < 45 :
    
    plt.subplot(15,3,i+1)
    plt.imshow(X_test[x], 'gray')
    plt.title('Real medic Image')
    plt.axis('off')
    
    plt.subplot(15,3,i+2)
    plt.imshow(y_test[x], 'gray')
    plt.title('Ground Truth Img')
    plt.axis('off')
    
    plt.subplot(15,3,i+3)
    plt.imshow(y_pred[x], 'gray')
    plt.title('Predicited Image')
    plt.axis('off')
    
    x += 1
    i += 3
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Classifier

# COMMAND ----------

info = [
    'benign'   ,  # 0
    'normal'   ,  # 1
    'malignant',  # 2
]

# COMMAND ----------

path = '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'

# COMMAND ----------

X = []
y = []

label_num = -1

for label_class in os.listdir(path) :
    
    new_path   = path + label_class
    label_num += 1
    
    for img in os.listdir(new_path) :
        if 'mask' not in img :
            
            y.append(label_num)
            x = cv2.imread(path + label_class +'/'+img, cv2.IMREAD_GRAYSCALE)
            X.append(img_to_array(Image.fromarray(cv2.resize(x, (128,128)))))

# COMMAND ----------

X = np.array(X)
y = np.array(y)

# COMMAND ----------

X/= 255.0

# COMMAND ----------

from keras.utils import to_categorical

# COMMAND ----------

y = to_categorical(y)

# COMMAND ----------

print(X.shape)
print(y.shape)

# COMMAND ----------

print(X.min())
print(X.max())

# COMMAND ----------

plt.imshow(X[0], 'gray')
plt.axis('off')

# COMMAND ----------

from keras.models import load_model

# COMMAND ----------

localize = load_model('./cancer_image_model.h5')

# COMMAND ----------

M = localize.predict(X)

# COMMAND ----------

print(M.min())
print(M.max())

plt.imshow(M[0], 'gray')
plt.axis('off')

# COMMAND ----------

# MAGIC %md
# MAGIC # Data distribution

# COMMAND ----------

import pandas
import seaborn

# COMMAND ----------

seaborn.histplot(data = pandas.DataFrame({'id' : [info[p] for p in np.argmax(y, axis = 1)]}), x = 'id')
plt.title('Distribution of classes accross the entire dataset', fontsize = 15)

# COMMAND ----------

# MAGIC %md
# MAGIC <div style = "text-align: justify">Although this is a imbalanced distribution, a model can easily be developed that does well in classification task. This is beause these images have clear distinctions among them.</div>

# COMMAND ----------

# MAGIC %md
# MAGIC # train-test split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(M, y, test_size = 0.1, shuffle = True, random_state = 1)

# COMMAND ----------

print(X_train.shape)
print(y_train.shape)

# COMMAND ----------

print(X_test.shape)
print(y_test.shape)

# COMMAND ----------

from numpy.random import randint

# COMMAND ----------

plt.figure(figsize = (20,20))
i = 0
SIZE = 702
while i < 25 :
    
    x = randint(0, SIZE)
    plt.subplot(5,5,i+1)
    plt.imshow(X_train[x], 'gray')
    plt.title(f'{info[np.argmax(y_train[x])]}', fontsize = 15)
    plt.axis('off')
    
    i += 1
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data augmentation

# COMMAND ----------

from keras.preprocessing.image import ImageDataGenerator

# COMMAND ----------

train_gen = ImageDataGenerator(horizontal_flip = True, rotation_range = 15, width_shift_range = [-10, 10], height_shift_range = [-10, 10], zoom_range = [0.80, 1.00])

# COMMAND ----------

train_gen.fit(X_train)

# COMMAND ----------

pointer = train_gen.flow(X_train, y_train)

# COMMAND ----------

trainX, trainy = pointer.next()

# COMMAND ----------

plt.figure(figsize = (20,20))

i = 0

while i < 25 :
    
    plt.subplot(5, 5, i+1)
    plt.imshow(trainX[i], 'gray')
    plt.title(f'{info[np.argmax(trainy[i])]}', fontsize = 15)
    plt.axis('off')
    
    i += 1
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # make CNN model

# COMMAND ----------

from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.layers import Flatten
from keras.layers import Dense

# COMMAND ----------

def conv_block (filterx) :
    
    model = Sequential()
    
    model.add(Conv2D(filterx, (3,3), strides = 1, padding = 'same', kernel_regularizer = 'l2'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(LeakyReLU())
    
    model.add(MaxPooling2D())
    
    return model

def dens_block (hiddenx) :
    
    model = Sequential()
    
    model.add(Dense(hiddenx, kernel_regularizer = 'l2'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(LeakyReLU())
    
    return model

# COMMAND ----------

def cnn (filter1, filter2, filter3, filter4, hidden1) :
    
    model = Sequential([
        
        Input((128,128,1,)),
        conv_block(filter1),
        conv_block(filter2),
        conv_block(filter3),
        conv_block(filter4),
        Flatten(),
        dens_block(hidden1),
        Dense(3, activation = 'softmax')
    ])
    
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.0005), metrics = ['accuracy'])
    
    return model

# COMMAND ----------

model = cnn(32, 64, 128, 256, 32)
model.summary()

# COMMAND ----------

from keras.utils import plot_model

# COMMAND ----------

plot_model(model, 'cancer_classify.png', show_shapes = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # fit()

# COMMAND ----------

checkp = ModelCheckpoint('./valid_classifier.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)

# COMMAND ----------

history = model.fit(train_gen.flow(X_train, y_train, batch_size = 64), epochs = 400, validation_data = (X_test, y_test), callbacks = [checkp])

# COMMAND ----------

plt.figure(figsize = (20,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training_loss', 'validation_loss'])
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('Loss val wrt. Epochs', fontsize = 15)

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictions

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# COMMAND ----------

model = keras.models.load_model('./valid_classifier.h5')

# COMMAND ----------

y_pred = model.predict(X_test)

# COMMAND ----------

y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)

# COMMAND ----------

print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred, target_names = info))

# COMMAND ----------

# MAGIC %md
# MAGIC # Confusion matrix

# COMMAND ----------

cm = confusion_matrix(y_test,y_pred)

# COMMAND ----------

plt.figure(figsize = (12,12))
ax = seaborn.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels = info, yticklabels = info)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

# COMMAND ----------

# MAGIC %md
# MAGIC # Overall task
# MAGIC <div style = "text-align: justify">Now that the models are complete, we first get the mask for input image and then classify the tumor type <b>benign, malignant or normal</b> based on mask shape.</div>

# COMMAND ----------

image_path = [
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/benign/benign (110).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/benign/benign (100).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/benign/benign (101).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/benign/benign (107).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/normal/normal (101).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/normal/normal (111).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/normal/normal (106).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant/malignant (115).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant/malignant (111).png',
    '../input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant/malignant (110).png',
]

# COMMAND ----------

from keras.models import load_model

# COMMAND ----------

# MAGIC %md
# MAGIC # load models

# COMMAND ----------

classifier = load_model('./valid_classifier.h5')
localize = load_model('./cancer_image_model.h5')

# COMMAND ----------

# MAGIC %md
# MAGIC # load images

# COMMAND ----------

testX = []
for img in image_path :
    testX.append(img_to_array(Image.fromarray(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (128,128)))))

# COMMAND ----------

testX = np.array(testX)
testX/= 255.0

# COMMAND ----------

print(testX.shape)
print(f'Minimum : {testX.min()}')
print(f'Maximum : {testX.max()}')

# COMMAND ----------

plt.imshow(testX[0], 'gray')
plt.axis('off')

# COMMAND ----------

# MAGIC %md
# MAGIC # predict mask and label

# COMMAND ----------

predY = localize.predict(testX)

# COMMAND ----------

print(predY.shape)

# COMMAND ----------

plt.imshow(predY[0], 'gray')
plt.axis('off')

# COMMAND ----------

print(predY.min())
print(predY.max())

# COMMAND ----------

pred_label = classifier.predict(predY)

# COMMAND ----------

print(np.argmax(pred_label, axis = 1))
plt.figure(figsize = (10,40))

i = 0
j = 0
while i < 20 :
    
    plt.subplot(10,2,i+1)
    plt.imshow (testX[j], 'gray')
    plt.title('Original Image', fontsize = 15)
    plt.axis('off')
    
    plt.subplot(10,2,i+2)
    plt.imshow (predY[j], 'gray')
    plt.title(f'{info[np.argmax(pred_label[j])]}', fontsize = 15)
    plt.axis('off')
    
    j += 1
    i += 2
plt.show()