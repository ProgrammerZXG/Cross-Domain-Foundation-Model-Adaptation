import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import glob
import torch

class BasicDataset(Dataset):

    def __init__(self,patch_h,patch_w,datasetName,netType,train_mode = False):

        self.patch_h = patch_h
        self.patch_w = patch_w

        if netType == 'unet' or netType == 'deeplabv3plus':
            self.imgTrans = False
        else: 
            self.imgTrans = True

        self.transform = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
        ])    

        self.dataset = datasetName

        if datasetName == 'seam':
            self.n1 = 1006
            self.n2 = 782
            # self.train_data_dir = '../data/seismicFace/train/input'
            # self.train_label_dir = '../data/seismicFace/train/target'
            # self.valid_data_dir = '../data/seismicFace/valid/input'
            # self.valid_label_dir = '../data/seismicFace/valid/target'
            self.train_data_dir = '/home/zxguo/data/seamai_1006x782/seamaiForTrain/input'
            self.train_label_dir = '/home/zxguo/data/seamai_1006x782/seamaiForTrain/target'
            self.valid_data_dir = '/home/zxguo/data/seamai_1006x782/seamaiForVal/input'
            self.valid_label_dir = '/home/zxguo/data/seamai_1006x782/seamaiForVal/target'
        elif datasetName == 'salt':
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = '../data/geobody/train/input'
            self.train_label_dir = '../data/geobody/train/target'
            self.valid_data_dir = '../data/geobody/valid/input'
            self.valid_label_dir = '../data/geobody/valid/target'
        elif datasetName == 'fault':
            self.n1 = 896
            self.n2 = 896
            self.train_data_dir = '../data/deepFault/train/image'
            self.train_label_dir = '../data/deepFault/train/label'
            self.valid_data_dir = '../data/deepFault/valid/image'
            self.valid_label_dir = '../data/deepFault/valid/label'
        elif datasetName == 'crater':
            self.n1 = 1022
            self.n2 = 1022
            self.train_data_dir = '../data/crater/train/image'
            self.train_label_dir = '../data/crater/train/label'
            self.valid_data_dir = '../data/crater/valid/image'
            self.valid_label_dir = '../data/crater/valid/label'
        elif datasetName == 'das':
            self.n1 = 512
            self.n2 = 512
            self.train_data_dir = '../data/das/train/image'
            self.train_label_dir = '../data/das/train/label'
            self.valid_data_dir = '../data/das/valid/image'
            self.valid_label_dir = '../data/das/valid/label'
        else:
            print("Dataset error!!")
        print('netType:' + netType)
        print('dataset:' + datasetName)
        print('patch_h:' + str(patch_h))
        print('patch_w:' + str(patch_w))

        if train_mode:
            self.data_dir = self.train_data_dir
            self.label_dir = self.train_label_dir
        else:
            self.data_dir = self.valid_data_dir
            self.label_dir = self.valid_label_dir

        self.ids = len(os.listdir(self.data_dir))
    def __len__(self):
        return self.ids

    def __getitem__(self,index):
        
        dPath = self.data_dir+'/'+str(index)+'.dat'
        tPath = self.label_dir+'/'+str(index)+'.dat'
        data = np.fromfile(dPath,np.float32).reshape(self.n1,self.n2)
        label = np.fromfile(tPath,np.int8).reshape(self.n1,self.n2)

        data = np.reshape(data,(1,1,self.n1,self.n2))
        data = np.concatenate([data,self.data_aug(data)],axis=0)
        label = np.reshape(label,(1,1,self.n1,self.n2))
        label = np.concatenate([label,self.data_aug(label)],axis=0)

        if self.imgTrans:
            img_tensor = np.zeros([2,1,self.patch_h*14,self.patch_w*14],np.float32)
            for i in range(data.shape[0]):
                img = Image.fromarray(np.uint8(data[i,0]))
                img_tensor[i,0] = self.transform(img)
            data = img_tensor
            data = data.repeat(3,axis=1)
        elif not self.imgTrans:
            data = data/255

        return data,label

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.fliplr(np.squeeze(data))
        return data_fliplr.reshape(b,c,h,w)

if __name__ == '__main__':

    train_set = BasicDataset(72,56,'seam','setr1',True,True)
    print(train_set.__getitem__(0)[1].shape)
    print(len(train_set))
