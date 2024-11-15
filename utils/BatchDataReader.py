import os
import natsort
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import numpy as np
import cv2
import random
from skimage import transform
import utils

class CubeDataset(Dataset):
    def __init__(self, data_dir,data_id,data_size,modality,
                 labelity,is_dataaug=True,is_norm=True):
        self.is_dataaug=is_dataaug
        self.is_norm=is_norm
        self.datanum=data_id[1]-data_id[0]
        self.modality=modality
        self.labelity=labelity
        self.data_size = data_size
        self.modalitynum=len(modality)
        self.datasetlist={'data':{},'label':{}}
        for modal in modality:
            self.datasetlist['data'].update({modal: {}})
            ctlist = os.listdir(os.path.join(data_dir, modal))
            ctlist = natsort.natsorted(ctlist)
            for ct in ctlist[data_id[0]:data_id[1]]:
                self.datasetlist['data'][modal].update({ct: {}})
                ctaddress = os.path.join(data_dir, modal, ct)
                self.datasetlist['data'][modal][ct] = ctaddress
        for labelid in labelity:
            ctlist = os.listdir(os.path.join(data_dir, labelid))
            ctlist = natsort.natsorted(ctlist)
            for ct in ctlist[data_id[0]:data_id[1]]:
                self.datasetlist['label'].update({ct: {}})
                labeladdress = os.path.join(data_dir, labelid, ct)
                self.datasetlist['label'][ct] = labeladdress

    def __getitem__(self, index):
        data = np.zeros((self.modalitynum,self.data_size[0],self.data_size[1],self.data_size[2]))
        label = np.zeros((1,self.data_size[1], self.data_size[2]))
        for i,modal in enumerate(self.modality):
            ct=list(self.datasetlist['data'][modal])[index]
            img=np.load(self.datasetlist['data'][modal][ct]).astype(np.float32)
            data[i,:,:,:]= img.transpose(1,2,0)

            # 加距离图，但其在投影方向的位置不完全固定
            for u in range(self.data_size[1]):
                for v in range(self.data_size[2]):
                    data[:, 0, u, v] = abs(u - 0.5 * self.data_size[1]) + abs(v - 0.5 * self.data_size[2])
        
        for modal in self.labelity:
            ct = list(self.datasetlist['label'])[index]
            label[0,:,:]=cv2.imread(self.datasetlist['label'][ct], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if self.is_dataaug==True:
            data,label=self.augmentation(data,label)
        # if self.is_norm==True:
        #     for i,modal in enumerate(self.modality):
        #         data[i]=utils.standardization(data[i])
        data = torch.from_numpy(np.ascontiguousarray(data))
        label = torch.from_numpy(np.ascontiguousarray(label))
        return data,label,ct

    def __len__(self):
        return self.datanum

    def augmentation(self,image,annotation):
        #horizontal flip
        if torch.randint(0, 4, (1,))==0:
            image=image[:,:,::-1,:]
            annotation=annotation[:,::-1,:]
        #Vertical flip
        if torch.randint(0, 4, (1,))==0:
            image = image[ :, :, :, ::-1]
            annotation = annotation[ :, :, ::-1]
        #rot90
        if torch.randint(0, 4, (1,))==0:
            k=torch.randint(1, 4, (1,))
            image=np.rot90(image,k,axes=(2,3))
            annotation = np.rot90(annotation, k, axes=(1, 2))
        #Translation in z-axis
        if torch.randint(0, 4, (1,))==0:
            dis=random.randint(1,20)
            if random.randint(0, 1) == 0:
                image=np.concatenate((image[:,-dis:,:,:],image[:,:-dis,:,:]),axis=1)
            else:
                image=np.concatenate((image[:,dis:,:,:],image[:,:dis,:,:]),axis=1)
            annotation = annotation
        return image,annotation
