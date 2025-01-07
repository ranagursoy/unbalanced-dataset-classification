import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from transformers import CLIPProcessor, CLIPModel


class BalancedSpermDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment_classes=None, downsample_classes=None, max_samples_per_class=None, upsample_factor=2):
        """
        Dengeli bir veri seti oluşturur. Veri artırımı (upsampling) ve azaltımı (downsampling) destekler.

        Args:
            data_dir (str): Veri seti dizini.
            transform (callable, optional): Görüntü dönüşüm işlemleri.
            augment_classes (list, optional): Up sampling yapılacak sınıf indeksleri.
            downsample_classes (list, optional): Down sampling yapılacak sınıf indeksleri.
            max_samples_per_class (int, optional): Down sampling için sınıf başına maksimum örnek sayısı.
            upsample_factor (int, optional): Up sampling faktörü.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augment_classes = augment_classes
        self.downsample_classes = downsample_classes
        self.max_samples_per_class = max_samples_per_class
        self.upsample_factor = upsample_factor
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))

        # Veri yükleme ve dengeleme
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                all_images = os.listdir(class_path)

                # Downsampling
                if self.downsample_classes and label in self.downsample_classes:
                    if len(all_images) > self.max_samples_per_class:
                        all_images = list(np.random.choice(all_images, self.max_samples_per_class, replace=False))

                # Upsampling
                if self.augment_classes and label in self.augment_classes:
                    upsampled_images = all_images * self.upsample_factor
                    all_images.extend(upsampled_images[:self.upsample_factor * len(all_images)])

                for img_name in all_images:
                    img_path = os.path.join(class_path, img_name)
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label
    
    def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    def get_classes(self):
        return self.classes


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, freeze_layers=0, activation=nn.ReLU):
        """
        ResNet50 modelini tanımlar. Dondurulacak katmanlar ve aktivasyon fonksiyonları dinamik olarak ayarlanabilir.

        Args:
            num_classes (int): Çıkış sınıf sayısı.
            dropout_rate (float): Dropout oranı.
            freeze_layers (int): Dondurulacak katman sayısı.
            activation (callable): Kullanılacak aktivasyon fonksiyonu.
        """
        super(ResNet50Classifier, self).__init__()  # Corrected this line
        self.resnet = models.resnet50(pretrained=True)

        # Katmanları dondurma
        for layer in list(self.resnet.children())[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Çıkış katmanı
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, num_classes),
            activation()
        )

    def forward(self, x):
        return self.resnet(x)


class VGG16Classifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, activation=nn.ReLU):
        """
        VGG16 modelini tanımlar. Dropout oranı ve aktivasyon fonksiyonları ayarlanabilir.

        Args:
            num_classes (int): Çıkış sınıf sayısı.
            dropout_rate (float): Dropout oranı.
            activation (callable): Kullanılacak aktivasyon fonksiyonu.
        """
        super(VGG16Classifier, self).__init__()  # Corrected this line
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vgg16.classifier[6].in_features, num_classes),
            activation()
        )

    def forward(self, x):
        return self.vgg16(x)


class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, weight1=0.5, weight2=0.5):
        """
        Ensemble modeli. İki farklı modeli verilen ağırlıklarla birleştirir.

        Args:
            model1 (nn.Module): Birinci model.
            model2 (nn.Module): İkinci model.
            weight1 (float): Birinci modelin ağırlığı.
            weight2 (float): İkinci modelin ağırlığı.
        """
        super(EnsembleModel, self).__init__()  # Corrected this line
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, x):
        output1 = self.model1(x)
        output2 = self.model2(x)
        return self.weight1 * output1 + self.weight2 * output2

class BiomedCLIPClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, activation=nn.ReLU, freeze_layers=0):
        super(BiomedCLIPClassifier, self).__init__()
        self.base_model = create_model("ViT-B-16", pretrained="openai")

        # Tüm katmanları dondur
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Fine-tuning için bazı katmanları aç
        if freeze_layers > 0:
            for name, param in list(self.base_model.named_parameters())[-freeze_layers:]:
                param.requires_grad = True

        # Tam bağlantı katmanı (FC Layer)
        self.fc = nn.Sequential(
            nn.Linear(self.base_model.visual.output_dim, 512),
            activation(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        self.text_descriptions = [
            "Normal",
            "AmorphHead",
            "DoubleHead",
            "NarrowAcrosome",
            "PyriformHead",
            "TaperedHead",
            "TwistedNeck",
            "AsymmetricNeck",
            "DoubleTail",
            "RoundHead",
            "ThickNeck",
            "TwistedTail",
            "CurlyTail",
            "LongTail",
            "PinHead",
            "ShortTail",
            "ThinNeck",
            "VacuolatedHead"
        ]
        self.text_features = self.base_model.encode_text(self.text_descriptions)

    def forward(self, x):
        # Görüntü özelliklerini çıkar
        image_features = self.base_model.encode_image(x)

        # Görüntü ve metin benzerliğini hesapla
        logits_per_image, logits_per_text = image_features @ self.text_features.T, self.text_features @ image_features.T

        # En yüksek benzerlik skoruna göre sınıflandırma
        return logits_per_image

def create_model(model_name="ViT-B-16", pretrained="openai"):
    """
    CLIP modelini yükler veya oluşturur.

    Args:
        model_name (str): Modelin ismi (örneğin ViT-B-16).
        pretrained (str): Önceden eğitilmiş model (örneğin openai).

    Returns:
        CLIPModel: Yüklenmiş model.
    """
    model = CLIPModel.from_pretrained(pretrained)
    processor = CLIPProcessor.from_pretrained(pretrained)
    return model

# Sınıf analiz fonksiyonu
def analyze_classes(dataset):
    """
    Veri setindeki her sınıfın örnek sayısını analiz eder.

    Args:
        dataset (BalancedSpermDataset): Analiz edilecek veri seti.

    Returns:
        dict: Her sınıfın örnek sayısı.
    """
    class_counts = {class_name: 0 for class_name in dataset.get_classes()}
    for _, label in dataset:
        class_counts[dataset.get_classes()[label]] += 1
    return class_counts


# Dinamik augment/downsample sınıflarını belirleme fonksiyonu
def determine_sampling_classes(class_counts, augment_threshold=500, downsample_threshold=1000):
    """
    Veri setindeki sınıf örnek sayısına göre augment ve downsample sınıflarını belirler.

    Args:
        class_counts (dict): Her sınıfın örnek sayısı.
        augment_threshold (int): Augment işlemi için eşik.
        downsample_threshold (int): Downsample işlemi için eşik.

    Returns:
        tuple: Augment edilecek sınıflar ve downsample edilecek sınıflar.
    """
    augment_classes = [i for i, count in enumerate(class_counts.values()) if count < augment_threshold]
    downsample_classes = [i for i, count in enumerate(class_counts.values()) if count > downsample_threshold]
    return augment_classes, downsample_classes