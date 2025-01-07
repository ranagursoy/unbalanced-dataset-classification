from collections import Counter
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


class SpermDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))  # Sınıflar alfabetik sıralanır
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        with Image.open(img_path) as img:
            img = img.convert('L')  # Gri tonlamalı
            if self.transform:
                img = self.transform(img)
        return img, label


data_dir = "train/train"  
dataset = SpermDataset(data_dir)
class_counts = Counter(label for _, label in dataset)

class_names = dataset.classes
for class_idx, count in class_counts.items():
    print(f"Class {class_idx} ({class_names[class_idx]}): {count} örnek")

sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
print("\nSınıf Dağılımı (Azdan Çoğa):")
for class_idx, count in sorted_classes:
    print(f"Class {class_idx} ({class_names[class_idx]}): {count} örnek")
