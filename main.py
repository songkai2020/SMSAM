import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold,train_test_split
from utils import ReadImgdata,dice_coef,dice_coef_loss,shuffle_data,Assign_weight,evaluation,Frozen_Weight
from model import SMSAM




if __name__=='__main__':

  train_path='/content/drive/MyDrive/train'
  test_path='/content/drive/MyDrive/test'
  mask='/content/drive/MyDrive/mask'
  entropy='/content/drive/MyDrive/entropy_glcm'
  layer_name=['CT1','conv11','conv12','CT2','conv13','conv14','CT3','conv15','conv16','CT4','conv17','conv18']
  tol_loss=[]
  tol_acc=[]
  count=0
  #prepare train data and train labels
  train_data,train_label,mask_label,trans=ReadImgdata(train_path,mask,entropy)
  train_label=np_utils.to_categorical(train_label,num_classes=2)
  #shuffle train data
  x_train,y_train,mask_train,trans=shuffle_data(train_data,train_label,mask_label,trans)
  
  #20-fold cross-validation
  Kflod = StratifiedKFold(n_splits=20,shuffle=False,random_state=0)
  for train,valid in Kflod.split(train_data,train_label.argmax(1)):
    #assign weight for each image
    sample_weights=Assign_weight(train,train_label)
    
    model = SMSAM((128,128,1))
    #load pre-trained weights
    model.load_weights('/content/drive/MyDrive/pretrain.h5',by_name=True)  
    model=Frozen_Weight(layer_name,model)
    #train
    adam=Adam(lr=1e-4,decay=1e-8)
    model.compile(optimizer=adam,loss={'seg':dice_coef_loss,'pred':'categorical_crossentropy'}, metrics = {'seg':dice_coef,'pred':'accuracy'})
    model.fit([x_train[train],trans[train]],[mask_train[train],y_train[train]],epochs=40,batch_size=32,
              validation_data=([x_train[valid],trans[valid]],[mask_train[valid],y_train[valid]]))
  
    #prepare test data and test labels
    test_data,test_label,mask_label_,trans_=ReadImgdata(test_path,mask,entropy)
    test_label=np_utils.to_categorical(test_label,num_classes=2)
    #shuffle test data
    test_train,test_label,mask_test,trans_=shuffle_data(test_data,test_label,mask_label_,trans_)

    _,_,loss,_,accuracy = model.evaluate([test_train,trans_],[mask_test,test_label])
    tol_loss.append(loss)
    tol_acc.append(accuracy)
    # accuracy1,sensitivity1,specificity1,auc1=evaluation(test_train,trans_,test_label,model)
    # print('sen:',sensitivity1)
    # print('spe:',specificity1)
    # print('auc:',auc1)
    print('test loss/accuracy:',loss,accuracy)
  print('avg_loss:',np.mean(np.array(tol_loss)),'avg_acc:',np.mean(np.array(tol_acc)))
  