'''
# 生成光密度数据
# OD = ln(max/gray)
'''

import os
import natsort
import numpy as np
from tqdm import tqdm

def gen_OD_npy(infile_path, outfile_path):
    ctlist = os.listdir(infile_path)
    ctlist = natsort.natsorted(ctlist)
    for ct in tqdm(ctlist):
        OCT_in = np.load(os.path.join(infile_path,ct))
        OD = np.where(OCT_in!=0, np.log(255/OCT_in).astype(np.float32), 0)
        np.save(os.path.join(outfile_path,ct),OD)

if __name__ == '__main__':
    infile_path='/public/home/G19830015/ycz/datasets/OCTA-500/3M/OCT_npy'
    outfile_path='/public/home/G19830015/ycz/datasets/OCTA-500/3M/OCT_OD_npy'
    gen_OD_npy(infile_path,outfile_path)

    infile_path='/public/home/G19830015/ycz/datasets/OCTA-500/6M/OCT_npy'
    outfile_path='/public/home/G19830015/ycz/datasets/OCTA-500/6M/OCT_OD_npy'
    gen_OD_npy(infile_path,outfile_path)