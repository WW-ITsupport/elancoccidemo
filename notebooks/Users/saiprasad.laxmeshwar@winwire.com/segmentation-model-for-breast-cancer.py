# Databricks notebook source
##########################################################################################################
#   For the task of segmentation we will be using only malignant and benign classes as the mask for      #
#   normal ultrasound will be zero always (probably!!)                                                   #
#                                                                                                        #
#   Model Used: U-Net                                                                                    #
##########################################################################################################

# COMMAND ----------

# MAGIC %fs ls abfss://elancoccicontainer@elancoccistorage.dfs.core.windows.net

# COMMAND ----------




# COMMAND ----------

configs = {"fs.azure.account.auth.type": "OAuth",
          "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
          "fs.azure.account.oauth2.client.id": "6edde992-5ac6-4df3-ad17-fc39ef7d9a0c",
          "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="ADLS",key="adls"),
          "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/bdcfaa46-3f69-4dfd-b3f7-c582bdfbb820/oauth2/token"}

# Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "abfss://elancoccicontainer@elancoccistorage.dfs.core.windows.net/",
  mount_point = "/mnt/ADLS",
  extra_configs = configs)

# COMMAND ----------

# MAGIC %md
# MAGIC # Dependencies

# COMMAND ----------

## Making essential imports
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
#import tensorflow as tf

# COMMAND ----------

#conecting to adls
spark.conf.set(
  "fs.azure.account.key.elancoccistorage.dfs.core.windows.net",
  "HQcink7eOaIQPLAufaPS+XepbGi2yRbR9s9XBQolhSqxcSqZibFMSFNOvTCr8ChppCIDk9MU7raCt8xmJnJ/fQ=="
)

# COMMAND ----------

#check directory list
display ( dbutils.fs.ls("abfss://elancoccicontainer@elancoccistorage.dfs.core.windows.net/Source/normal/") )

# COMMAND ----------

df = dbutils.fs.ls("mnt/ADLS/Source/normal/")
df

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Loader

# COMMAND ----------

## defining a frame for image and mask storage
framObjTrain = {'img' : [],
           'mask' : []
          }

## defining data Loader function
def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 256):
    imgNames = os.listdir(imgPath)
    
    names = []
    maskNames = []
    unames = []
    
    for i in range(len(imgNames)):
        unames.append(imgNames[i].split(')')[0])
    
    unames = list(set(unames))
    
    for i in range(len(unames)):
        names.append(unames[i]+').png')
        maskNames.append(unames[i]+')_mask.png')
    
    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'
    
    for i in range (len(names)):
        img = plt.imread(imgAddr + names[i])
        mask = plt.imread(maskAddr + maskNames[i])
        
        img = cv2.resize(img, (shape, shape)) 
        mask = cv2.resize(mask, (shape, shape))
        
        frameObj['img'].append(img)
        frameObj['mask'].append(mask)
        
    return frameObj

# COMMAND ----------

normalimagesdf = spark.read.format("image").load("abfss://elancoccicontainer@elancoccistorage.dfs.core.windows.net/Source/benign/")
display(normalimagesdf )

# COMMAND ----------

path = 'dbfs:/FileStore/tables/Dataset_BUSI_with_GT'
dir_list = [os.path.join(path,i) for i in os.listdir(path)]
size_dict = {}
for i,value in enumerate(dir_list):
    size_dict[os.listdir(path)[i]] = len(os.listdir(value))
size_dict 

# COMMAND ----------

# loading benign samples

framObjTrain = LoadData( framObjTrain, imgPath = "abfss://elancoccicontainer@elancoccistorage.dfs.core.windows.net/Source/normal"
                        , maskPath = "abfss://elancoccicontainer@elancoccistorage.dfs.core.windows.net/Source/normal"
                         , shape = 256)

# COMMAND ----------

# loading malignant samples

framObjTrain = LoadData( framObjTrain, imgPath = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant'
                        , maskPath = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant'
                         , shape = 256)

# COMMAND ----------

## displaying data loaded by our function
plt.figure(figsize = (10, 7))
plt.subplot(1,2,1)
plt.imshow(framObjTrain['img'][1])
plt.title('Ultra Sound Image')
plt.subplot(1,2,2)
plt.imshow(framObjTrain['mask'][1])
plt.title('Mask for Cancer')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Defination

# COMMAND ----------

###########################################################################
#                                Model   Defination                       #
###########################################################################


# this block essentially performs 2 convolution

def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    #first Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
    
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x =tf.keras.layers.Activation('relu')(x)
    
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    return x


# Now defining Unet 
def GiveMeUnet(inputImage, numFilters = 16, droupouts = 0.1, doBatchNorm = True):
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)
    
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)
    
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)
    
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)
    
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(droupouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(droupouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(droupouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(droupouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    return model

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Instantiation

# COMMAND ----------

## instanctiating model
inputs = tf.keras.layers.Input((256, 256, 3))
myTransformer = GiveMeUnet(inputs, droupouts= 0.07)
myTransformer.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# COMMAND ----------

# MAGIC %md
# MAGIC # Training For 50 Epochs

# COMMAND ----------

retVal = myTransformer.fit(np.array(framObjTrain['img']), np.array(framObjTrain['mask']), epochs = 50, verbose = 0)

# COMMAND ----------

plt.plot(retVal.history['loss'], label = 'training_loss')
plt.plot(retVal.history['accuracy'], label = 'training_accuracy')
plt.legend()
plt.grid(True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Testing

# COMMAND ----------

def predict16 (valMap, model, shape = 256):
    ## getting and proccessing val data
    img = valMap['img'][0:16]
    mask = valMap['mask'][0:16]
    #mask = mask[0:16]
    
    imgProc = img [0:16]
    imgProc = np.array(img)
    
    predictions = model.predict(imgProc)
  

    return predictions, imgProc, mask


def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(9,9))
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title(' image')
    
    plt.subplot(1,3,2)
    plt.imshow(predMask)
    plt.title('Predicted mask')
    
    plt.subplot(1,3,3)
    plt.imshow(groundTruth)
    plt.title('Actual mask')

# COMMAND ----------

sixteenPrediction, actuals, masks = predict16(framObjTrain, myTransformer)
Plotter(actuals[1], sixteenPrediction[1][:,:,0], masks[1])

# COMMAND ----------

Plotter(actuals[2], sixteenPrediction[2][:,:,0], masks[2])

# COMMAND ----------

Plotter(actuals[3], sixteenPrediction[3][:,:,0], masks[3])

# COMMAND ----------

Plotter(actuals[5], sixteenPrediction[5][:,:,0], masks[5])

# COMMAND ----------

Plotter(actuals[7], sixteenPrediction[7][:,:,0], masks[7])

# COMMAND ----------

Plotter(actuals[8], sixteenPrediction[8][:,:,0], masks[8])

# COMMAND ----------

Plotter(actuals[9], sixteenPrediction[9][:,:,0], masks[9])

# COMMAND ----------

Plotter(actuals[10], sixteenPrediction[10][:,:,0], masks[10])

# COMMAND ----------

# MAGIC %md
# MAGIC # Saving Model

# COMMAND ----------

myTransformer.save('BreastCancerSegmentor.h5')