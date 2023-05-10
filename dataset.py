import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T 
from sklearn.model_selection import StratifiedKFold

class dataset(data.Dataset):
    def __init__(self, train=False, val=False, all=False, test=False, kfold=0, aug=True):
        
        #path
        if train or val or all:
            imgPath = 'data/1. Original Images/a. Training Set/'
            gtPath = 'data/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv'
        else: #test
            imgPath = 'data/1. Original Images/b. Testing Set/'
        
        # prepare dataset
        self.imgs = [] #List[List[path,label,name]]
        if test:
            pathList = os.listdir(imgPath)
            pathList.sort(key=lambda x:int(x.split('.')[0]))
            for name in pathList:
                self.imgs.append([imgPath+name,-1,name])
        elif all:
            csvFile = pd.read_csv(gtPath)
            for _, row in csvFile.iterrows():
                name = row['image name']
                label = int(row['image quality level'])
                self.imgs.append([imgPath+name, label, name])
        elif train or val:
            csvFile = pd.read_csv(gtPath)
            labels = []
            imgList = []
            for _, row in csvFile.iterrows():
                name = row['image name']
                label = int(row['image quality level'])
                labels.append(label)
                imgList.append([imgPath+name, label, name])
            skf = StratifiedKFold(n_splits=5) #no need for shuffle
            for index, (train_index, val_index) in enumerate(skf.split(np.zeros_like(labels),labels)):
                if index == kfold:
                    break
            if train:
                for i in train_index:
                    self.imgs.append(imgList[i])
            else:
                for i in val_index:
                    self.imgs.append(imgList[i])

        # transform
        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
        }
        if train and aug:
            self.transform = T.Compose([
                T.Resize((420,420)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop(
                    size=((224,224)),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),

                T.ColorJitter(
                    brightness=data_aug['brightness'],
                    contrast=data_aug['contrast'],
                ),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])
        # used for incepv3
        # if train and aug:
        #     self.transform = T.Compose([
        #         T.Resize((640,640)),
        #         T.RandomHorizontalFlip(),
        #         T.RandomVerticalFlip(),
        #         T.RandomResizedCrop(
        #             size=((512,512)),
        #             scale=data_aug['scale'],
        #             ratio=data_aug['ratio']
        #         ),

        #         T.ColorJitter(
        #             brightness=data_aug['brightness'],
        #             contrast=data_aug['contrast'],
        #         ),
        #         T.ToTensor(),
        #         T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        #     ])
        # else:
        #     self.transform = T.Compose([
        #         T.Resize((512,512)),
        #         T.ToTensor(),
        #         T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        #     ])
        

    def __getitem__(self, index):
        path, label, name = self.imgs[index]
        img = Image.open(path).convert('RGB') # original pic has only one channel
        # or load by opencv which may cause minor difference
        # img = cv2.cvtColor(cv2.imread('data/2.jpg'), cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label, name
    
    def __len__(self):
        return len(self.imgs)


# test dataset
if __name__ == '__main__':
    ds1 = dataset(train=True,kfold=4)
    # ds2 = dataset(val=True,kfold=1)
    # ds3 = dataset(val=True,kfold=2)
    # ds4 = dataset(val=True,kfold=3)
    nameList1 = []
    # nameList2 = []
    # nameList3 = []
    # nameList4 = []
    nameList5 = []
    ds5 = dataset(val=True,kfold=4)
    length = len(ds5)
    for i in range(length):
        img, label, name = ds5.__getitem__(i)
        nameList5.append(label)
    print("asdasdsadasad")
    length = len(ds1)
    for i in range(length):
        img, label, name = ds1.__getitem__(i)
        nameList1.append(label)
    print(nameList5)
    print(nameList1)
    # length = len(ds1)
    # nameList1 = []
    # nameList2 = []
    # nameList3 = []
    # nameList4 = []
    # nameList5 = []
    # for i in range(length):
    #     img, label, name = ds1.__getitem__(i)
    #     nameList1.append(name)
    # for i in range(length):
    #     img, label, name = ds2.__getitem__(i)
    #     nameList2.append(name)
    # for i in range(length):
    #     img, label, name = ds3.__getitem__(i)
    #     nameList3.append(name)
    # for i in range(length):
    #     img, label, name = ds4.__getitem__(i)
    #     nameList4.append(name)
    # for i in range(length):
    #     img, label, name = ds5.__getitem__(i)
    #     nameList5.append(name)
    # set_c = set(nameList1) & set(nameList2) & set(nameList3) & set(nameList4) & set(nameList5)
    # print(set_c)
    # ds2 = dataset(train=True,kfold=0)
    # for i in range(len(ds1)):
    #     img, label, name = ds1.__getitem__(i)
    #     print(name)
    #     img, label, name = ds2.__getitem__(i)
    #     print(name)
    #     if i == 10:
    #         break
