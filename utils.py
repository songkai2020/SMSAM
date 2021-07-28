import os
import cv2
import numpy as np
import keras.backend as K
from sklearn.metrics import confusion_matrix,roc_auc_score


#load data and labels
def ReadImgdata(path,maskpath,path2):
  '''path : The path of original images.
  maskpath : The path of original masks.
  path2 : The path of original entropy matrixs.
  '''
  img_dataset=[]
  label_dataset=[]
  nii_data=[]
  Trans=[]
  file_folder = os.listdir(path)
  for img in file_folder:
    #image
    img_=cv2.imread(path+'/'+img,-1)
    img_=cv2.resize(img_,(128,128),interpolation=cv2.INTER_LINEAR)
    img_=img_[:,:,np.newaxis]
    label_dataset.append(img.split('_')[1])
    img_dataset.append(img_)
    #mask
    img_b = cv2.imread(maskpath+'/'+img.split('a')[0]+'a_Merge_'+img.split('a')[1],0)
    img_b=cv2.resize(img_b,(128,128),interpolation=cv2.INTER_LINEAR)
    img_b=img_b[:,:,np.newaxis]
    nii_data.append(img_b)


    #img_t=np.loadtxt(path2+'/'+img.split('.')[0]+'.txt')

    img_t=np.load(path2+'/'+img.split('.')[0]+'.npy')

    img_t=img_t[:,:,np.newaxis]
    Trans.append(img_t)
  img_dataset=np.array(img_dataset).astype('float32')
  nii_data = np.array(nii_data).astype('float32')
  Trans =np.array(Trans).astype('float32')
  #归一化
  img_dataset/=255.
  nii_data/=255.
  Trans/=255.
  label_dataset = np.array(label_dataset).astype('float32')
  return img_dataset,label_dataset,nii_data,Trans

#shuffle the data
def shuffle_data(data,label,mask,entropy):
  index = [i for i in range(len(label))]
  np.random.shuffle(index)
  x = data[index]
  y = label[index]
  mask = mask[index]
  entropy=entropy[index]
  return x,y,mask,entropy

#assign weight for each image
def Assign_weight(train,train_label):
  sample_weights=[]
  for sw in train:
    # print(train_label[sw])
    if np.argmax(train_label[sw])==0:
      sample_weights.append(1)
    if np.argmax(train_label[sw])==1:
      sample_weights.append(2)
  sample_weights=np.array(sample_weights)
  return sample_weights

#Calculate the Sensitivity,Specificity,AUC
def evaluation(test_train,trans_,test_label,model):
  labels=[]
  y_true=[]
  label1=[]
  for i in range(len(test_train)):
    test=test_train[i][np.newaxis,:,:,:]
    test_=trans_[i][np.newaxis,:,:,:]
    label_=model.predict([test,test_])
    label=np.argmax(label_[1])
    labels.append(label)
    y_true.append(np.argmax(test_label[i]))
    label1.append(label_[1][0][1])
  tn1, fp1, fn1, tp1 = confusion_matrix(y_true,labels).ravel()
  sensitivity1 = tp1/(tp1+fn1)
  specificity1 = tn1/(tn1+fp1)
  auc1 = roc_auc_score(y_true,label1)
  accuracy1 = (tp1+tn1)/(tp1+tn1+fp1+fn1)
  return accuracy1,sensitivity1,specificity1,auc1
 
#Frozen Weights of layers
def Frozen_Weight(layer_name,model):
  for name in layer_name:
    model.get_layer(name).trainable=False
  return model

 #Dice
def dice_coef(y_true, y_pred,smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)