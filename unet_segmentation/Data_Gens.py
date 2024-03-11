# -*- coding: utf-8 -*-

        #### Import All libraies used for training  #####
        
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import cv2
import PIL.Image as Image
from torchvision import transforms
import matplotlib.pyplot as plt

  ### Data_Generators ########
  
NUM_WORKERS=0
PIN_MEMORY=True
height=224
width=224

train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        #transforms.ColorJitter(brightness=0.005, contrast=0.005, saturation=0.005, hue=0.005),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1), 
        transforms.ColorJitter(saturation=0.1),
        transforms.ColorJitter(hue=0.1)]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.8946, 0.8659, 0.8638), (0.1050, 0.1188, 0.1180))      
    ])

test_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        transforms.ToTensor(),
        transforms.Normalize((0.8946, 0.8659, 0.8638), (0.1050, 0.1188, 0.1180))      
    ])


class Dataset_Train(Dataset):
    
    def __init__(self, image_dir, mask_dir, transforms):
        self.image_dir = image_dir
        
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transforms
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'.jpg')
        
        image = Image.open(img_path)
        image_tensor = self.transform(image)

        mask=cv2.imread(mask_path,0)
        mask[np.where(mask!=0)]=1
        mask=np.expand_dims(mask, axis=0)

        return image_tensor, mask, self.images[index][:-4]
    
    
def Data_Loader_Train( image_dir, mask_dir, train_transform, batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_Train( image_dir=image_dir, mask_dir=mask_dir, transforms=train_transform)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


class Dataset_Val(Dataset):
    
    def __init__(self, image_dir, mask_dir, transforms):
        self.image_dir = image_dir
        
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transforms
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'.jpg')
        
        image = Image.open(img_path)
        image_tensor = self.transform(image)

        mask=cv2.imread(mask_path,0)
        mask[np.where(mask!=0)]=1
        mask=np.expand_dims(mask, axis=0)

        return image_tensor, mask, self.images[index][:-4]
    
def Data_Loader_Val( image_dir,mask_dir, test_transform, batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_Val(image_dir=image_dir, mask_dir=mask_dir, transforms=test_transform)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


#train_imgs = r'C:\My_Data\Amaya\Train\img'
#train_masks = r'C:\My_Data\Amaya\Train\GT'
#
#val_loader = Data_Loader_Val(train_imgs, train_masks, test_transform, batch_size = 2)
#train_loader = Data_Loader_Train(train_imgs, train_masks, train_transform, batch_size = 2)
#
#a = iter(train_loader)
#a1 =next(a)
#img = a1[0][0,:].numpy()
#img = img.transpose(2,1,0)
#gt = a1[1][0,0,:].numpy()
#plt.figure()
#plt.imshow(img)
#plt.figure()
#plt.imshow(gt