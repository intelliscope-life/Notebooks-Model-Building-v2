import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import random
from get_test import get_csv_names

#Default Settings
Dataset_path = ''

Ignore_Blanks = True
Read_Type = 0 


images = []

#Default Picture Crop Points
s_x = 82;
e_x = 544;
s_y = 43;
e_y = 315;

size_x = e_x - s_x ;
size_y = e_y - s_y;

n = 0 
n_cat = 3;

print(size_y,size_x)

def is_blank(x):
    img = cv.imread(Dataset_path + x, 0)
    
    if np.sum(img[150:200,350:400])==637500:
        return True
    else:
        return False

def set_ignore_blanks(x):
    global Ignore_Blanks
    Ignore_Blanks = x

def set_dataset_path(path,readtype,ignore_folder=None):
    global Read_Type
    Read_Type = readtype
    
    global Dataset_path
    Dataset_path = path
    
    global images
    form = '.tiff'
    images = sorted(os.listdir(Dataset_path))
    images = [image for image in images if form in image]
    print(" 💡 => Total images ("+ form +") found ",len(images))
    
    if(ignore_folder!=None):
        images = [image for image in images if ignore_folder not in image]
        print(" ⚡️ => After ignoring folder loaded ",len(images))
    
    if(Ignore_Blanks):
        images = [image for image in images if is_blank(image)==0]
        print(" ⚡️ => After ignoring blanks loaded ",len(images))

    global n
    n = len(images)
    
    cat_2_count = 0
    
    for i in range(0,n):
        if images[i][-6]=='N':
            cat_2_count = cat_2_count + 1
        elif images[i][-6]=='A' or images[i][-6]=='O': #other or Abnormal
            cat_2_count = cat_2_count + 1
            
    n = cat_2_count
    
    return True

def show_i(x):
    img = cv.imread(Dataset_path + images[x], 0)
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.imshow(img[s_y:e_y,s_x:e_x])
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.show()
    
def show_img(x):
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.show()
    
X = None
Y = None
Y = None



def load_hybrid_data(leave_test = False, test_set=None):
    global s_x
    global e_x
    global s_y
    global e_y

    height = e_y - s_y;
    width  = e_x - s_x
    
    ecg_path  = Dataset_path[:-4] + 'ecg/'
    ecg_files = os.listdir(ecg_path)
    
    test_records =get_csv_names('test_records.csv', ecg_path, Dataset_path) # 'other_records.csv/ test_records.csv'

    global images
    images = [image for image in images if image in ecg_files]
    
    if(leave_test or test_set!=None):
        if test_set=='testset':
            images = [image for image in images if image in test_records]
        else:
            images = [image for image in images if image not in test_records]
    

    global n
    n = len(images)
    
    m = Read_Type*3 + int(Read_Type==0)
    e = Read_Type*6 + int(Read_Type==0)*2
    
    X = np.zeros(shape=(n,height,width,e),dtype=np.uint8)
    Y = np.zeros(shape=(n))
    Y = Y.astype('int8') 
    
    for i in range(0,n):
        
        im1 = cv.imread(Dataset_path + images[i], Read_Type)
        im2 = cv.imread(ecg_path     + images[i], Read_Type)
        
        im1 = im1[s_y:e_y,s_x:e_x]
        im2 = im2[s_y:e_y,s_x:e_x]
        
        im1 = im1.reshape((height,width,m))
        im2 = im2.reshape((height,width,m))
        
        X[i,:,:,0:m] = im1
        X[i,:,:,m:e] = im2

        if images[i][-6]=='N':
            Y[i] = 0
        else:
            Y[i] = 1

        if(i%1000==0):
            print(i)       

    print(str(n)+ " Images loaded across " + '2' + " Categories")  
    
    return n,X,Y,images   


def load_data(train_percentage):
    X = None
    Y = None
    
    if(Read_Type==1):
        X = np.zeros(shape=(n,272,462,3),dtype=np.uint8)
        Y = np.zeros(shape=(n))
        Y = Y.astype('int8') 
    elif(Read_Type==0):
        X = np.zeros(shape=(n,272,462),dtype=np.uint8)
        Y = np.zeros(shape=(n))
        Y = Y.astype('int8')
            
    if(len(Dataset_path)==0):
        print("Dataset path not set")
        return 
    
    for i in range(0,n):
        img = cv.imread(Dataset_path + images[i], Read_Type)
        
        
        X[i]= img[s_y:e_y,s_x:e_x]
        
        
        

        if images[i][-6]=='N':
            Y[i] = 0
        else:
            Y[i] = 1

        if(i%1000==0):
            print(i)

    print(str(n)+ " Images loaded across " + '2' + " Categories")  
    
    return n,X,Y

def select_records(x):
    global images
    records = np.array(pd.read_csv('REFERENCE.csv'))[:,]  

    abnormal_records = []
    normal_records   = []

    for i in range(0,len(records)):
        if(records[i][1]==-1):
            normal_records.append(records[i][0])
        else:
            abnormal_records.append(records[i][0])

    random.seed(x)
    a_records = []
    for i in range(0,len(images)):
        rec = images[i][0:5]
        if(rec in abnormal_records and rec not in a_records ):
            a_records.append(rec)
            
    n_records = []
    for i in range(0,len(images)):
        rec = images[i][0:5]
        if(rec in normal_records and rec not in n_records):
            n_records.append(rec)        
    
    print(len(a_records)," In abnormal population. Selecting ",len(n_records)," out of them 👌")
    
    abnormal_records_selected = random.sample(a_records, k=len(n_records))
    
    return n_records,abnormal_records_selected


def load_balanced_hybrid_data(leave_test = False, test_set=None):
    global s_x
    global e_x
    global s_y
    global e_y

    height = e_y - s_y;
    width  = e_x - s_x
    
    ecg_path  = Dataset_path[:-4] + 'ecg/'
    ecg_files = os.listdir(ecg_path)
    
    test_records =get_csv_names('test_records.csv', ecg_path, Dataset_path) # 'other_records.csv/ test_records.csv'

    global images
    images = [image for image in images if image in ecg_files]
    
    if(leave_test or test_set!=None):
        if test_set=='testset':
            images = [image for image in images if image in test_records]
        else:
            images = [image for image in images if image not in test_records]
    
    n_r,a_r = select_records(120)
    global n
    n = len(images)
    
    s_images = images[:]

    count = 0 
    for i in range(0,len(images)):
        rec = images[i][0:5]
        if(not(rec in n_r or rec in a_r)):
            s_images.remove(images[i])
            count = count + 1
    print(count," abnormal ones excluded 🥺")
    n = len(s_images) #replace n with real value

    
    m = Read_Type*3 + int(Read_Type==0)
    e = Read_Type*6 + int(Read_Type==0)*2
    
    X = np.zeros(shape=(n,height,width,e),dtype=np.uint8)
    Y = np.zeros(shape=(n))
    Y = Y.astype('int8') 
    
    for i in range(0,n):
        im1 = cv.imread(Dataset_path + s_images[i], Read_Type)
        im2 = cv.imread(ecg_path     + s_images[i], Read_Type)
        
        im1 = im1[s_y:e_y,s_x:e_x]
        im2 = im2[s_y:e_y,s_x:e_x]
        
        im1 = im1.reshape((height,width,m))
        im2 = im2.reshape((height,width,m))
        
        X[i,:,:,0:m] = im1
        X[i,:,:,m:e] = im2

        if s_images[i][-6]=='N':
            Y[i] = 0
        else:
            Y[i] = 1

        if(i%1000==0):
            print(i)       

    print(str(n)+ " Images loaded across " + '2' + " Categories")  
    
    return n,X,Y,s_images   

