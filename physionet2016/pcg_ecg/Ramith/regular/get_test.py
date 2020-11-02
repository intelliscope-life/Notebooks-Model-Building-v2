import numpy as np
import pandas as pd
import os

def get_csv_names(csv_name, ecg_dir, pcg_dir):
    def get_img_names(ecg_dir, pcg_dir):
        names=[]
        for file in os.listdir(pcg_dir): # does not matter whether ecg_dir/ pcg_dir is choosen
            if ('.tiff' in file) and (file in os.listdir(ecg_dir)):names.append(file)
        names = sorted(names)
        return names

    all_names=get_img_names(ecg_dir, pcg_dir)
    
    records = np.array(pd.read_csv(csv_name))[:,1]  
    names   = []
    for record in records:
        for name in all_names:
            if record in name:
                names.append(name)
                
    sorted(names)
    return names