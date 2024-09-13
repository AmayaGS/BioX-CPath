# -*- coding: utf-8 -*-

import os.path
from PIL import Image
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch.utils.data import Dataset


class histoDataset(Dataset):

    def __init__(self, df, transform, label):

        self.transform = transform
        self.labels = df[label].astype(int).tolist()
        self.filepaths = df['File_location'].tolist()
        self.patient_IDs = df['Patient_ID'].tolist()
        self.filenames = df['Filename'].tolist()
        self.patch_names = df['Patch_name'].tolist()
        self.coordinates = df['Patch_coordinates'].tolist()
        self.stain_types = df['Stain_type'].tolist()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):

        try:

            image = Image.open(self.filepaths[idx])
            # If the image has an alpha channel, remove it
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            patient_id = self.patient_IDs[idx]
            filename = self.filenames[idx]
            patch_name = self.patch_names[idx]
            coordinate = self.coordinates[idx]
            self.image_tensor = self.transform(image)
            self.image_label = self.labels[idx]
            stain_type = self.stain_types[idx]

            return self.image_tensor, self.image_label, patient_id, filename, patch_name, coordinate, stain_type

        except (FileNotFoundError, IndexError):
            return self.__getitem__(idx)


class Loaders:

        def slides_dataloader(self, df, ids, transform, slide_batch, num_workers, shuffle, collate, label, patient_id):

            # TRAIN dict
            patient_subsets = {}

            for i, file in enumerate(ids):
                new_key = f'{file}'
                patient_subset = histoDataset(df[df[patient_id] == file], transform, label)
                patient_subsets[new_key] = torch.utils.data.DataLoader(patient_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False, collate_fn=collate)

            return patient_subsets