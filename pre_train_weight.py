from keras.layers import *
from keras.models import Model
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.optimizers import Adam
import keras.backend as K
from sklearn.model_selection import StratifiedKFold,train_test_split
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix,roc_auc_score
from MSAM import MSSAM,MSCAM

def ReadImgdata(path,maskpath):
  '''path : The path of original images.
  maskpath : The path of original masks.
  '''
  img_dataset=[]
  nii_data=[]
  file_folder = os.listdir(path)
  for img in file_folder:
    #image
    img_=cv2.imread(path+'/'+img,-1)
    img_=cv2.resize(img_,(128,128),interpolation=cv2.INTER_LINEAR)
    img_=img_[:,:,np.newaxis]

    img_dataset.append(img_)
    #mask
    img_b = cv2.imread(maskpath+'/'+img.split('a')[0]+'a_Merge_'+img.split('a')[1],0)
    img_b=cv2.resize(img_b,(128,128),interpolation=cv2.INTER_LINEAR)
    img_b=img_b[:,:,np.newaxis]
    nii_data.append(img_b)
    
  img_dataset=np.array(img_dataset).astype('float32')
  nii_data = np.array(nii_data).astype('float32')
  #归一化
  img_dataset/=255.
  nii_data/=255.
  return img_dataset,nii_data



def Novel_U_net():
  model_input=Input((128,128,1))
  conv1 = Conv2D(96, (5, 5), padding='same',name='conv1',dilation_rate=2)(model_input)
  conv1 = BatchNormalization(name='bn1')(conv1)
  conv1 = Activation('relu',name='ac1')(conv1)
  conv1 = Conv2D(96, (5, 5),padding='same',name='conv2',dilation_rate=2)(conv1)
  conv1 = BatchNormalization(name='bn2')(conv1)
  conv1 = Activation('relu',name='ac2')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2),name='pool1')(conv1)

  conv2 = Conv2D(256, (3, 3), padding='same',name='conv3',dilation_rate=2)(pool1)
  Conv2 = BatchNormalization(name='bn3')(conv2)
  conv2 = Activation('relu',name='ac3')(conv2)
  conv2 = Conv2D(256, (3, 3),padding='same',name='conv4',dilation_rate=2)(conv2)
  conv2 = BatchNormalization(name='bn4')(conv2)
  conv2 = Activation('relu',name='ac4')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2),name='pool2')(conv2)

  conv3 = Conv2D(384, (3, 3), padding='same',name='conv5')(pool2)
  conv3 = BatchNormalization(name='bn5')(conv3)
  conv3 = Activation('relu',name='ac5')(conv3)
  conv3 = Conv2D(384, (3, 3), padding='same',name='conv6')(conv3)
  conv3 = BatchNormalization(name='bn6')(conv3)
  conv3 = Activation('relu',name='ac6')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2),name='pool3')(conv3)

  conv4 = Conv2D(384, (3, 3), padding='same',name='conv7')(pool3)
  conv4 = BatchNormalization(name='bn7')(conv4)
  conv4 = Activation('relu',name='ac7')(conv4)
  conv4 = Conv2D(256, (3, 3), padding='same',name='conv8')(conv4)
  conv4 = BatchNormalization(name='bn8')(conv4)
  conv4 = Activation('relu',name='ac8')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2),name='pool4')(conv4)

  conv5 = Conv2D(256, (3, 3), activation='relu', padding='same',name='conv9')(pool4)
  conv5 = BatchNormalization(name='bn9')(conv5)
  conv5 = Activation('relu',name='ac9')(conv5)
  conv5 = Conv2D(256, (3, 3), activation='relu', padding='same',name='conv10')(conv5)
  conv5 = BatchNormalization(name='bn10')(conv5)
  conv5 = Activation('relu',name='ac10')(conv5)

  conv5_2 = MSSAM(name='mssam')([conv5,conv4,conv3])
  conv5_1 = MSCAM(name='mscam')([conv5,conv4,conv3])
  conv5=concatenate([conv5_1,conv5_2],name='concat1',axis=-1)

  #decoder
  up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',name='CT1')(conv5), conv4], axis=-1,name='concat2')
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',name='conv11')(up6)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',name='conv12')(conv6)

  up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',name='CT2')(conv6), conv3], axis=-1,name='concat3')
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',name='conv13')(up7)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',name='conv14')(conv7)

  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',name='CT3')(conv7), conv2], axis=-1,name='concat4')
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',name='conv15')(up8)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',name='conv16')(conv8)

  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',name='CT4')(conv8), conv1], axis=-1,name='concat5')
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',name='conv17')(up9)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',name='conv18')(conv9)

  Seg = Conv2D(1, (1, 1), activation='sigmoid',name='seg')(conv9)
  model=Model(inputs=[model_input],outputs=[Seg])
  return model

def dice_coef(y_true, y_pred,smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


if __name__=='__main__':
  image_path='/content/drive/MyDrive/test'
  mask_path='/content/drive/MyDrive/mask'
  image,mask=ReadImgdata(image_path,mask_path)
  model=Novel_U_net()
  adam=Adam(lr=1e-4,decay=1e-8)
  model.compile(optimizer=adam,loss={'seg':dice_coef_loss}, metrics = {'seg':dice_coef})
  model.fit(image,mask,epochs=40,batch_size=32,validation_split=0.1)
  model.save_weights('/content/drive/MyDrive/pretrain.h5')