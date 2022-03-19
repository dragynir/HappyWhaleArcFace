from torch.utils.data import Dataset
import cv2
import torch
import pandas as pd
import os

class HappyWhaleDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['label'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }

class HappyWhaleTestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transforms=None,
        return_labels=True,
    ):
        self.df = df
        self.images = self.df["image"]
        self.image_dir = image_dir
        self.transforms = transforms
        self.return_labels = return_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images.iloc[idx])
        if not os.path.exists(image_path):
            return self[idx+1]

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.return_labels:
            label = self.df['label'].iloc[idx]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img
