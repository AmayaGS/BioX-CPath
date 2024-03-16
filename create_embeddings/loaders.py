# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:15:48 2023

@author: AmayaGS
"""


import os, os.path
from PIL import Image
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch.utils.data import Dataset


class histoDataset(Dataset):

    def __init__(self, df, transform, label):

        self.transform = transform
        self.labels = df[label].astype(int).tolist()
        self.filepaths = df['File_location'].tolist()
        self.patient_ID = df['Patient_ID'].tolist()
        self.file = df['Filename'].tolist()
        self.patch_name = df['Patch_name'].tolist()
        self.coordinates = df['Patch_coordinates'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        try:

            image = Image.open(self.filepaths[idx])
            # If the image has an alpha channel, remove it
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            patient_id = self.patient_ID[idx]
            file = self.file[idx]
            patch_name = self.patch_name[idx]
            coordinates = self.coordinates[idx]
            self.image_tensor = self.transform(image)
            self.image_label = self.labels[idx]

            return self.image_tensor, self.image_label, patient_id, file, patch_name, coordinates

        except FileNotFoundError:
            return None


class Loaders:

        def slides_dataloader(self, df, ids, transform, slide_batch, num_workers, shuffle, collate, label='Pathotype_binary', patient_id="Patient_ID"):

            # TRAIN dict
            patient_subsets = {}

            for i, file in enumerate(ids):
                new_key = f'{file}'
                patient_subset = histoDataset(df[df[patient_id] == file], transform, label=label)
    #            if len(train_subset) != 0:
                patient_subsets[new_key] = torch.utils.data.DataLoader(patient_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False, collate_fn=collate)

            return patient_subsets