# SMSAM
## Segmentation-based Multi-scale attention model for KRAS mutation prediction in rectal cancer
## 1.Introduction
We propose a Segmentation-based Multi-scale attention model(SMSAM) to predict the mutation status of KRAS gene in rectal cancer patients with limited t2-weighted Magnetic Resonance Imaging (MRI) data. This is repository of Segmentation-based Multi-scale attention model.

## 2.Requirement  
Tensorflow='1.15.2'  
Keras='2.3.1'  
h5py=='2.10'  

## 3.Training process  
1.Execute ROI.py and glcm.py to obtain the patches and the corresponding entropy matrixs.  
2.Execute pre_train_weight.py to get the pre-trained weights.  
3.Execute main.py to train SMSAM.  
