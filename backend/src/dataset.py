import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PeriodontalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, apply_clahe=True):
        """
        Args:
            csv_file (str): Path to the csv file with annotations (train.csv, val.csv, test.csv).
            root_dir (str): Directory with all the images (e.g., '../data').
            transform (callable, optional): Optional transform to be applied on a sample.
            apply_clahe (bool): Whether to apply CLAHE preprocessing for radiograph enhancement.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.apply_clahe = apply_clahe
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]['File name']
        level = self.data_frame.iloc[idx]['Level']
        
        # Load from flat images directory (no level subfolders)
        img_path = os.path.join(self.root_dir, "images", img_name)
        
        # Read image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        if self.apply_clahe:
            image = self.clahe.apply(image)
            
        # Convert grayscale to RGB since most pretrained models expect 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
            
        # Extract and normalize tabular features
        age = float(self.data_frame.iloc[idx]['Age']) / 100.0
        gender = float(self.data_frame.iloc[idx]['Gender']) / 2.0
        missing_teeth = float(self.data_frame.iloc[idx]['Number of missing teeth']) / 32.0
        implant = float(self.data_frame.iloc[idx]['Implant'])
        residual_root = float(self.data_frame.iloc[idx]['Residual root'])
        func_tooth_log = float(self.data_frame.iloc[idx]['Functional tooth logarithm']) / 32.0
        
        tabular_features = torch.tensor([age, gender, missing_teeth, implant, residual_root, func_tooth_log], dtype=torch.float32)

        # Map levels {1, 2, 3} to 0-indexed classes {0, 1, 2}
        label = int(level) - 1

        return image, tabular_features, label

def get_transforms(img_size=224):
    """
    Returns train and validation transforms.
    Uses medically reasonable augmentations (no aggressive zooming or warping).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Increased rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Added translation and scaling
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Slightly increased jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_heavy_transforms(img_size=224):
    """
    Returns heavy train transforms for Experiment 3.
    Includes slightly more aggressive but clinically reasonable variations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25), # More rotation
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10), # Added shear, more translation/scaling
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Simulate slight machine misalignments
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), # Simulate varying X-ray clarity
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
