import os
import itertools
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model_dataset import BalancedSpermDataset, ResNet50Classifier, EnsembleModel
import timm

# create_model tanımı
def create_model(model_name, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained)

# BiomedCLIPClassifier tanımı
class BiomedCLIPClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, activation=torch.nn.ReLU, freeze_layers=0):
        super(BiomedCLIPClassifier, self).__init__()
        self.base_model = create_model("vit_base_patch16_224", pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False

        if freeze_layers > 0:
            for name, param in list(self.base_model.named_parameters())[-freeze_layers:]:
                param.requires_grad = True

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.base_model.head.in_features, 512),
            activation(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model.forward_features(x)
        pooled_features = features[:, 0]  # cls_token kullanılarak tek bir vektör alınır
        return self.fc(pooled_features)

# Çıkış klasörü
output_dir = "grid_search_results_final-2"
os.makedirs(output_dir, exist_ok=True)

# Veri seti yolu
data_dir = "train2"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sınıfları analiz etmek için fonksiyon
def analyze_classes(dataset):
    class_counts = {class_name: 0 for class_name in dataset.get_classes()}
    for _, label in dataset:
        class_counts[dataset.get_classes()[label]] += 1
    return class_counts

# Dinamik olarak augment ve downsample sınıflarını belirleme
def determine_sampling_classes(class_counts, augment_threshold=500, downsample_threshold=1000):
    augment_classes = [i for i, count in enumerate(class_counts.values()) if count < augment_threshold]
    downsample_classes = [i for i, count in enumerate(class_counts.values()) if count > downsample_threshold]
    return augment_classes, downsample_classes

# Veri seti ve sınıf analizi
dataset = BalancedSpermDataset(data_dir, transform=transform)
class_counts = analyze_classes(dataset)
print("Sınıf Analizi:", class_counts)

param_grid = {
    "optimizer": [torch.optim.Adam],
    "learning_rate": [0.0001],
    "dropout_rate": [0.2],
    "freeze_layers": [0],
    "batch_size": [32],
    "epochs": [50],
    "activation": [torch.nn.LeakyReLU],
    "weights": [(0.7, 0.3)],
    "augment_threshold": [500],
    "downsample_threshold": [800]
}
grid_combinations = list(itertools.product(*param_grid.values()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_index = 0

for i, params in enumerate(grid_combinations):
    if i < start_index:
        continue
    (optimizer_fn, lr, dropout, freeze, batch_size, epochs, activation, 
     (weight1, weight2), augment_threshold, downsample_threshold) = params

    print(f"Configuration {i+1}/{len(grid_combinations)}: {params}")

    augment_classes, downsample_classes = determine_sampling_classes(
        class_counts,
        augment_threshold=augment_threshold,
        downsample_threshold=downsample_threshold
    )

    dataset = BalancedSpermDataset(
        data_dir=data_dir,
        transform=transform,
        augment_classes=augment_classes,
        downsample_classes=downsample_classes,
        max_samples_per_class=downsample_threshold,
        upsample_factor=2
    )

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    resnet_model = ResNet50Classifier(
        num_classes=len(dataset.get_classes()),
        dropout_rate=dropout,
        freeze_layers=freeze,
        activation=activation
    ).to(device)

    biomedclip_model = BiomedCLIPClassifier(
        num_classes=len(dataset.get_classes()),
        dropout_rate=dropout,
        activation=activation,
        freeze_layers=freeze
    ).to(device)

    ensemble_model = EnsembleModel(
        resnet_model,
        biomedclip_model,
        weight1=weight1,
        weight2=weight2
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_fn(ensemble_model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    patience = 5
    best_val_loss = float('inf')
    counter = 0

    def save_metrics(loader, dataset_name):
        preds, labels = [], []
        with torch.no_grad():
            for images, lbls in loader:
                images, lbls = images.to(device), lbls.to(device)
                outputs = ensemble_model(images)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(lbls.cpu().numpy())
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"{dataset_name} Confusion Matrix")
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png"))
        plt.close()

        report = classification_report(labels, preds, target_names=dataset.get_classes())
        with open(os.path.join(output_dir, f"{dataset_name}_classification_report.txt"), "w") as f:
            f.write(report)

    for epoch in range(epochs):
        ensemble_model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = ensemble_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        save_metrics(train_loader, "train")
        save_metrics(val_loader, "validation")
        
        torch.save(ensemble_model.state_dict(), os.path.join(output_dir, f"ensemble_model_{i+1}.pth"))

    
