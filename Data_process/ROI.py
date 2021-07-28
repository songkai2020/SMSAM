import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import cv2
from PIL import Image
import imageio
import os
import csv
import pydicom
from pydicom import dcmread


def Save_img(input_path, save_path):
    #Slice the mask
    files = os.listdir(input_path)
    for img in files:
        file_folder = os.path.basename(img).split('.')[0]
        if os.path.exists(save_path + '/' + file_folder):
            continue
        else:
            os.mkdir(save_path + '/' + file_folder)
        img_load = nib.load(input_path + '/' + img)
        img_fdata = img_load.get_fdata()
        total = img_fdata.shape[-1]
        for index in range(total):
            img_fdata_ = img_fdata[:, :, index]
            # Swap pixels
            img_index = np.array(np.swapaxes(img_fdata_, 0, 1))
            ret, binary = cv2.threshold(img_index, 0, 255, cv2.THRESH_BINARY)
            # Save Image
            imageio.imwrite(save_path + '/' + file_folder + '/' + file_folder + '_' + str(index) + '_' + '.png', binary)


def Rect_img(path, csv_writer):
    # Get the ROI
    black = np.zeros((512, 512))
    file_folder = os.listdir(path)
    for folder in file_folder:
        imgs = os.listdir(path + '/' + folder)
        for file in imgs:
            binary = cv2.imread(path + '/' + folder + '/' + file, -1)
            # img = cv2.imread(path+'/'+folder+'/'+file)
            if file == '.ipynb_checkpoints':  
                continue
            else:
                if binary.any() == black.any():

                    pass
                else:
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    location = [x, y, w, h]
                    # dict[count]=location
                    ID = file.split('h')[0]
                    num = file.split('_')[2]

                    csv_writer.writerow([ID, num, x, y, w, h])


def data_csv(imgpath, savepath):
    # Save coordinate information of ROI
    f = open(savepath + '/' + 'datainfo.csv', 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['ID', 'num', 'x', 'y', 'w', 'h'])
    Rect_img(imgpath, csv_writer)
    f.close()


def Read_csv(path):
    # Read coordinate information of ROI
    file = open(path, 'r')
    rows = csv.reader(file)
    coordinate = []
    for row in rows:
        coordinate.append(row)
    return coordinate


def Read_CT(csv_path, img_path, save_path):
    # Extract patches
    location = Read_csv(csv_path)
    
    file_folder = os.listdir(img_path)
    for file in file_folder:
        
        sub_folder = os.listdir(img_path + '/' + file)
        for sub_file in sub_folder:
            if os.path.exists(save_path + '/' + sub_file):
                pass
            else:
                os.mkdir(save_path + '/' + sub_file)
            
            id_ = sub_file.split('h')[0]
            
            for i in range(1, len(location)):
               
                Id = location[i][0]
                if Id == id_:
                    
                    print(img_path + '/' + file + '/' + sub_file + '/' + 'IM' + str(location[i][1]))
                    dataset = dcmread(img_path + '/' + file + '/' + sub_file + '/' + 'IM' + str(location[i][1]))
                    x = int(location[i][2])
                    w = int(location[i][4])
                    y = int(location[i][3])
                    h = int(location[i][5])
               
                    cx = (x + x + w) // 2
                    cy = (y + y + h) // 2
                    print(cx, cy)
                    x = cx - 71
                    y = cy - 71
                    new_img = dataset.pixel_array[y:(y + 142), x:(x + 142)]
                    new_img = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    # print('y_-y',y_-y,'w',w)
                    imageio.imwrite(save_path + '/' + sub_file + '/' + sub_file + location[i][1] + '.png', new_img)
                    # if int(w)>int(h):
                    #   y_1=(y+y+h)//2
                    #   y=y_1-w//2
                    #   y_=y_1+w//2
                    #   if (y_-y)<10 or w<10:
                    #     continue
                    #   else:
                    #     new_img=dataset.pixel_array[y:y_,x:(x+w)]
                    #     # print('y_-y',y_-y,'w',w)
                    #     imageio.imwrite(save_path+'/'+sub_file+'/'+sub_file+location[i][1]+'.png',new_img)
                    #     # # cv2.imwrite(save_path+'/'+sub_file+'/'+sub_file+location[i][1]+'.png',new_img)
                    # else:
                    #   x_1=(x+x+w)//2
                    #   x=x_1-h//2
                    #   x_=x_1+h//2
                    #   if (x_-x)<10 or h<10:
                    #     continue
                    #   else:
                    #     new_img=dataset.pixel_array[y:(y+h),x:x_]
                    #     # cv2.imwrite(save_path+'/'+sub_file+'/'+sub_file+location[i][1]+'.png',new_img)
                    #     imageio.imwrite(save_path+'/'+sub_file+'/'+sub_file+location[i][1]+'.png',new_img)


def sort_dicom(img_path, save_path):
    # Sort dicom
    file_folder = os.listdir(img_path)

    for file in file_folder:
      
        sub_folder = os.listdir(img_path + '/' + file)
        for sub_file in sub_folder:
            
            if os.path.exists(save_path + '/' + file + '/' + sub_file) or sub_file == '.ipynb_checkpoints' or \
                    sub_file.split('h')[1] != 'rT2a':
                pass
            else:
                os.mkdir(save_path + '/' + file)
                os.mkdir(save_path + '/' + file + '/' + sub_file)
            
            if os.path.basename(sub_file).split('h')[1] == 'rT2a':
                dic = os.listdir(img_path + '/' + file + '/' + sub_file)
                # print(img_path+'/'+file+'/'+sub_file)
                dict = {}
                
                for i in range(len(dic)):
                    ds = pydicom.read_file(img_path + '/' + file + '/' + sub_file + '/' + 'IM' + str(i))
                    dict['IM' + str(i)] = float(ds.SliceLocation)
                    
                dict_sort = sorted(dict.items(), key=lambda x: x[1])
               
                count = 0
                for item in dict_sort:
                    
                    os.rename(img_path + '/' + file + '/' + sub_file + '/' + item[0],
                              save_path + '/' + file + '/' + sub_file + '/IM' + str(count))
                    
                    count += 1



def rect_ROI(nii_path,nii_slice_save,csv_save,dicom_path,sort_path,rec_save_path):
    Save_img(nii_path,nii_slice_save)
    data_csv(nii_slice_save,csv_save)
    sort_dicom(dicom_path,sort_path)
    csv_path=csv_save+'/datainfo.csv'
    Read_CT(csv_path,sort_path,rec_save_path)
    print('Finish!')


if __name__ == "__main__":
    nii_path = r'G:\recta_cancer_T2\ROI_T2'  # The path of original masks.
    nii_slice_save = r'G:\recta_cancer_T2\nii_slice'  # The path of sliced masks.
    csv_save = r'G:\recta_cancer_T2'  # The path to store the coordinate information of ROI.
    dicom_path = r'G:\recta_cancer_T2\T2_case1-200'  # The path of original images(Dicom).
    sort_path = r'G:\recta_cancer_T2\T2_case1-200_sorted'  # The path to store the sorted Dicom.
    rec_save_path = r'G:\recta_cancer_T2\rec_save'  # The path to store the new patches.
    rect_ROI(nii_path, nii_slice_save, csv_save, dicom_path, sort_path, rec_save_path)

