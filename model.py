from keras.layers import *
from keras.models import Model
from MSAM import MSSAM,MSCAM


def SMSAM(input_shape):
  model_input=Input(input_shape)
  model_input2=Input((142,142,1))
  #encoder
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

  #attention module
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
  

  #entropy branch
  Y = Convolution2D(filters=96,kernel_size=(11,11),strides=(4,4),padding='same',activation='relu')(model_input2)
  Y = BatchNormalization()(Y)
  Y = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(Y)
  Y = Convolution2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')(Y)
  Y = BatchNormalization()(Y)
  Y = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(Y)
 

  conv_c1=Conv2D(64,(3,3),activation='relu',padding='same',strides=(2,2))(conv9)

  conv_c2=Conv2D(128,(3,3),activation='relu',padding='same',strides=(2,2))(conv_c1)

  conv_c3=Conv2D(256,(3,3),activation='relu',padding='same',strides=(2,2))(conv_c2)

  conv_c4=Conv2D(256,(3,3),activation='relu',padding='same',strides=(2,2),name='conv_c4')(conv_c3)

  X=concatenate([conv5,conv_c4,Y],axis=-1)

  X = GlobalAveragePooling2D()(X)
  X= Dense(2,activation='softmax',name='pred')(X)
  model=Model(inputs=[model_input,model_input2],outputs=[Seg,X])
  return model