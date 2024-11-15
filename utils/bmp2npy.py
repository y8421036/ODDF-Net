import cv2
import numpy as np
import os
import natsort
from PIL import Image
from skimage import transform
from tqdm import tqdm
import utils


def gen_npy_from_bmp(bmpfile_path, npzfile_path, size):
    ctlist = os.listdir(bmpfile_path)
    ctlist = natsort.natsorted(ctlist)
    for ct in ctlist:
        data = []
        bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
        bscanlist = natsort.natsorted(bscanlist)
        for bscan in bscanlist:
            data.append(np.array(Image.open(os.path.join(bmpfile_path,ct,bscan)).resize(size))) # 改变图像大小
        np.save(os.path.join(npzfile_path,ct),data)

def gen_bmp_from_npy(npy_path, out_path, size):
    npy = np.load(npy_path)
    npy = utils.normalization(npy)*255
    shape = npy.shape
    for i in tqdm(range(shape[0])):
        img = np.array(npy[i])
        # img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(out_path, str(1001+i)+'.bmp'), img)

if __name__ == '__main__':
    bmpfile_path='/data2/datasets/OCTA-500/6M/OCT'
    npzfile_path='/data2/datasets/OCTA-500/6M/OCT_npy'
    size = (400,128)
    gen_npy_from_bmp(bmpfile_path,npzfile_path,size)

    bmpfile_path='/data2/datasets/OCTA-500/6M/OCTA'
    npzfile_path='/data2/datasets/OCTA-500/6M/OCTA_npy'
    size = (400,128)
    gen_npy_from_bmp(bmpfile_path,npzfile_path,size)

    bmpfile_path='/data2/datasets/OCTA-500/3M/OCT'
    npzfile_path='/data2/datasets/OCTA-500/3M/OCT_npy'
    size = (304, 128)
    gen_npy_from_bmp(bmpfile_path,npzfile_path,size)

    bmpfile_path='/data2/datasets/OCTA-500/3M/OCTA'
    npzfile_path='/data2/datasets/OCTA-500/3M/OCTA_npy'
    size = (304, 128)
    gen_npy_from_bmp(bmpfile_path,npzfile_path,size)
    
    # gen_bmp_from_npy('/data2/datasets/OCTA-500/6M/OCTA_npy/10001.npy','logs/OCTA_10001', (400,640))