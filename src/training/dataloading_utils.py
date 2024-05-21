import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os 
import pandas as pd
import cv2
import numpy as np
import albumentations as A

DEBUG = True
DATASET_PATH = 'dataset'
BATCH_SIZE = 32

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.Resize(224, 224)
])

def load_data(data_folder_path: str, debug: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
    '''
    Load the data from the dataset folder.
    
    Args:  
    
    `data_folder_path`: The path to the dataset folder.
    `debug`: If True, print some debug information.
    '''
    filenames = list(filter(lambda x: 'csv' in x, os.listdir(data_folder_path)))
    X,Y = [], []
    
    for filename in filenames:
        targets = pd.read_csv(os.path.join(data_folder_path, filename)).to_numpy()
        frame_indices = targets[:, 0].astype(int)
        Y.append(targets[:, 1:])
        
        # Load images from the corresponding folders
        folder = filename.removesuffix('.csv')
        images = []
        for frame_idx in frame_indices:
            fram_idx_pad = str(frame_idx).zfill(4)
            img_name = f'{fram_idx_pad}.jpg'
            
            if os.path.exists(os.path.join(data_folder_path, folder, img_name)):
                img = cv2.imread(os.path.join(data_folder_path, folder, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                print(f'File {img_name} not found in {os.path.join(data_folder_path, folder)}')
                continue
            
            augmented = transform(image=img)
            img = augmented['image']
            
            # Move the channel axis to the first dimension
            img = np.moveaxis(img, -1, 0)
            images.append(img)
            
        X.append(images)
        
    if debug:
        print(f'Number of distinct runs in data: {len(X)}')
        print(f'Run lengths in frames (dataset size before augmentations): {[len(run) for run in X]}')
        print(f'Dimension of the first run: {len(X[0])} images')
        print(f'Dimension of the first image: {X[0][0].shape}')
        print(f'Number of distinct runs in labels: {len(Y)}')
        print(f'Dimension of the first label: {Y[0][0].shape}')
        
    return X, Y

def train_test_split(data: list[np.ndarray], labels: list[np.ndarray], test_size: float = 0.2, debug: bool = True) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    '''
    Split the data into training and testing sets.
    
    Args:
    
    `data`: The data to split.
    `labels`: The labels to split.
    `test_size`: The proportion of the data to be used for testing.
    `debug`: If True, print some debug information.
    '''
    split_idx = int(len(data) * (1 - test_size))
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # Flatten the data
    train_data = np.array([img for run in train_data for img in run])
    test_data = np.array([img for run in test_data for img in run])
    train_labels = np.array([label for run in train_labels for label in run])
    test_labels = np.array([label for run in test_labels for label in run])
    
    
    if debug:
        print(f'Training data shape: {train_data.shape}, training labels shape: {train_labels.shape}')
        print(f'Testing data shape: {test_data.shape}, testing labels shape: {test_labels.shape}')
        
    return train_data, test_data, train_labels, test_labels

def create_dataloader(data: np.ndarray, labels: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    '''
    Create a DataLoader object from the data and labels.
    
    Args:
    
    `data`: The data to load.
    `labels`: The labels to load.
    `batch_size`: The size of the batch.
    `shuffle`: If True, shuffle the data.
    '''
    x = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    