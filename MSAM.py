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


class MSSAM(Layer):
  def __init__(self,**kwargs):
    super(MSSAM,self).__init__(**kwargs)

  def build(self,input_shape):
    #create learnable Weight1,Weight2,Weight2
    self.weights1=self.add_weight(
        name='weight_trainable3',
        shape=(1,),
        dtype=None,
        initializer='uniform',
        regularizer=None,
        trainable=True,
        constraint=None)
    
    self.weights2=self.add_weight(
        name='weight_trainable4',
        shape=(1,),
        dtype=None,
        initializer='uniform',
        regularizer=None,
        trainable=True,
        constraint=None)
    
    self.weights3=self.add_weight(
        name='weight_trainable5',
        shape=(1,),
        dtype=None,
        initializer='uniform',
        regularizer=None,
        trainable=True,
        constraint=None)
    self.built = True

  def compute_output_shape(self,input_shape):
    input_shape1=input_shape[0]
    input_shape2=input_shape[1]
    input_shape3=input_shape[2]
    return input_shape1
  
  def call(self,input):
    F_map1=input[0]
    F_map2=input[1]
    F_map3=input[2]

    shapes1=F_map1.get_shape().as_list()
    shapes2=F_map2.get_shape().as_list()
    shapes3=F_map3.get_shape().as_list()


    #Feature map M
    F_map=F_map1_c=F_map1=Convolution2D(filters=shapes1[-1],kernel_size=(1,1),strides=(1,1),padding='same',dilation_rate=2)(F_map1) #
    F_map1=Reshape((shapes1[-2]*shapes1[-3],shapes1[-1]))(F_map1)
    F_map1_c=Reshape((shapes1[-2]*shapes1[-3],shapes1[-1]))(F_map1)
    F_map1_c=tf.transpose(F_map1_c,[0,2,1])
    F_map1_att=K.batch_dot(F_map1,F_map1_c)
    
    #Feature map M
    F_map2_c=F_map2=F_map1=Convolution2D(filters=shapes1[-1],kernel_size=(1,1),strides=(1,1),padding='same',dilation_rate=2)(F_map2)
    F_map2=Reshape((shapes2[-2]*shapes2[-3],shapes1[-1]))(F_map2)
    F_map2_c=Reshape((shapes2[-2]*shapes2[-3],shapes1[-1]))(F_map2)
    F_map2_c=tf.transpose(F_map2_c,[0,2,1])
    F_map2_att=K.batch_dot(F_map2,F_map2_c)
    
    #Feature map L
    F_map3_c=F_map3=F_map1=Convolution2D(filters=shapes1[-1],kernel_size=(1,1),strides=(1,1),padding='same',dilation_rate=2)(F_map3)
    F_map3=Reshape((shapes3[-2]*shapes3[-3],shapes1[-1]))(F_map3)
    F_map3_c=Reshape((shapes3[-2]*shapes3[-3],shapes1[-1]))(F_map3)
    F_map3_c=tf.transpose(F_map3_c,[0,2,1])
    F_map3_att=K.batch_dot(F_map3,F_map3_c)
    
    #bilinear downsampling
    F_map2_att=tf.expand_dims(F_map2_att, -1)
    F_map2_att=tf.image.resize_images(F_map2_att, [shapes1[-3]*shapes1[-2],shapes1[-3]*shapes1[-2]],0)
    F_map2_att=tf.squeeze(F_map2_att,axis=[-1],name=None,squeeze_dims=None)
    
    F_map3_att=tf.expand_dims(F_map3_att, -1)
    F_map3_att=tf.image.resize_images(F_map3_att, [shapes1[-3]*shapes1[-2],shapes1[-3]*shapes1[-2]],0)
    F_map3_att=tf.squeeze(F_map3_att,axis=[-1],name=None,squeeze_dims=None)
    

    F_map_att=self.weights1*F_map1_att+self.weights2*F_map2_att+self.weights3*F_map3_att
    F_map_att=Activation('sigmoid')(F_map_att)
    #weighted sum
    F_map_=K.batch_dot(F_map_att,Reshape((shapes1[-3]*shapes1[-2],shapes1[-1]))(F_map))
    F_map=F_map+Reshape((shapes1[-3],shapes1[-2],shapes1[-1]))(F_map_)
    return F_map


class MSCAM(Layer):
  def __init__(self,**kwargs):
    super(MSCAM,self).__init__(**kwargs)

  def build(self,input_shape):
    #create learnable Weight1,Weight2,Weight2
    self.weights1=self.add_weight(
        name='weight_trainable1',
        shape=(1,),
        dtype=None,
        initializer='uniform',
        regularizer=None,
        trainable=True,
        constraint=None)
    
    self.weights2=self.add_weight(
        name='weight_trainable2',
        shape=(1,),
        dtype=None,
        initializer='uniform',
        regularizer=None,
        trainable=True,
        constraint=None)
    
    self.weights3=self.add_weight(
        name='weight_trainable6',
        shape=(1,),
        dtype=None,
        initializer='uniform',
        regularizer=None,
        trainable=True,
        constraint=None)
    
    self.built = True
  def compute_output_shape(self,input_shape):
    input_shape1=input_shape[0]
    input_shape2=input_shape[1]
    input_shape3=input_shape[2]
    return input_shape1
  
  def call(self,input):
    F_map=F_map1=input[0]
    F_map2=input[1]
    F_map3=input[2]
    
    shapes1= F_map1.get_shape().as_list()
    shapes2= F_map2.get_shape().as_list()
    shapes3= F_map3.get_shape().as_list()
    #Feature map S
    F_map1_c=F_map1
    F_map1=Reshape((shapes1[-2]*shapes1[-3],shapes1[-1]))(F_map1)
    F_map1_c=Reshape((shapes1[-2]*shapes1[-3],shapes1[-1]))(F_map1)
    F_map1_c=tf.transpose(F_map1_c,[0,2,1])
    F_map1_att=K.batch_dot(F_map1_c,F_map1)
    
    #Feature map M
    F_map2_c=F_map2=Convolution2D(filters=shapes1[-1],kernel_size=(1,1),strides=(1,1),padding='same')(F_map2)
    F_map2=Reshape((shapes2[-2]*shapes2[-3],shapes1[-1]))(F_map2)
    F_map2_c=Reshape((shapes2[-2]*shapes2[-3],shapes1[-1]))(F_map2)
    F_map2_c=tf.transpose(F_map2_c,[0,2,1])
    F_map2_att=K.batch_dot(F_map2_c,F_map2)
    
    #Feature map L
    F_map3_c=F_map3=Convolution2D(filters=shapes1[-1],kernel_size=(1,1),strides=(1,1),padding='same')(F_map3)
    F_map3=Reshape((shapes3[-2]*shapes3[-3],shapes1[-1]))(F_map3)
    F_map3_c=Reshape((shapes3[-2]*shapes3[-3],shapes1[-1]))(F_map3)
    F_map3_c=tf.transpose(F_map3_c,[0,2,1])
    F_map3_att=K.batch_dot(F_map3_c,F_map3)
    
    #weighted sum
    F_map_att=self.weights1*F_map1_att+self.weights2*F_map2_att+self.weights3*F_map3_att
    F_map_att=Activation('sigmoid')(F_map_att)

    F_map=F_map+K.batch_dot(F_map,F_map_att)

    return F_map
