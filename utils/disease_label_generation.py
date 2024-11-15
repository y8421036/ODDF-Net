import pandas as pd
import numpy as np

def gen_disease_label_npy(in_path, out_path):
    disease = pd.read_excel(io=in_path,usecols='A,E') #读取特定列
    disease = np.array(disease) # shape为[300,2]
    disease[disease=='NORMAL'] = 0
    disease[disease=='AMD'] = 1
    disease[disease=='DR'] = 2
    disease[disease=='CNV'] = 3
    disease[disease=='CSC'] = 4
    disease[disease=='RVO'] = 5
    disease[disease=='OTHERS'] = 6
    np.save(out_path,disease)

if __name__ == '__main__':
    in_path = '/data2/datasets/OCTA-500/3M/3M-Text labels.xlsx'
    out_path = '/data2/datasets/OCTA-500/3M/3M-Text labels.npy'
    gen_disease_label_npy(in_path,out_path)

    in_path = '/data2/datasets/OCTA-500/6M/6M-Text labels.xlsx'
    out_path = '/data2/datasets/OCTA-500/6M/6M-Text labels.npy'
    gen_disease_label_npy(in_path,out_path)