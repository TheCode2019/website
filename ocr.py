
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage import io, transform as sk_transform
import numpy as np
from PIL import Image

class Chars74KDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for dp, _, filenames in os.walk(directory):
            for f in filenames:
                if f.endswith('.png') or f.endswith('.jpg'):
                    full_path = os.path.join(dp, f)
                    label = self.extract_label(dp)
                    if label is not None:
                        self.image_paths.append(full_path)
                        self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images with labels.")

    def extract_label(self, directory_path):
        match = re.search(r'\d+', os.path.basename(directory_path))
        if match:
            return int(match.group()) - 1
        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Utilisation de PIL pour la conversion en grayscale
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Transformation pour les images
transformations = transforms.Compose([
    transforms.Resize((28, 28)),  # convertir en 28x28
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),  # convertir en tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalisation
])

class Enhanced_OCR_CNN(nn.Module):
    def __init__(self):
        super(Enhanced_OCR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 62)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout(x.view(-1, 64 * 7 * 7))
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x




