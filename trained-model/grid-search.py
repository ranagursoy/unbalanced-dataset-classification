import os
import itertools
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model_dataset import BalancedSpermDataset, ResNet50Classifier, BiomedCLIPClassifier, EnsembleModel
import timm

# Global configurations
OUTPUT_DIR = "grid_search_results_final-2"
DATA_DIR = "train2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create model function
def create_model(model_name, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained)

# Dataset class analysis
def analyze_classes(dataset):
    class_counts = {class_name: 0 for class_name in dataset.get_classes()}
    for _, label in dataset:
        class_counts[dataset.get_classes()[label]] += 1
    return class_counts

# Sampling determination
def determine_sampling_classes(class_counts, augment_threshold=500, downsample_threshold=1000):
    augment_classes = [i for i, count in enumerate(class_counts.values()) if count < augment_threshold]
    downsample_classes = [i for i, count in enumerate(class_counts.values()) if count > downsample_threshold]
    return augment_classes, downsample_classes

# Load dataset with augment/downsample

def load_dataset(data_dir, transform, augment_classes=None, downsample_classes=None, downsample_threshold=800):
    return BalancedSpermDataset(
        data_dir=data_dir,
        transform=transform,
        augment_classes=augment_classes,
        downsample_classes=downsample_classes,
        max_samples_per_class=downsample_threshold,
        upsample_factor=2
    )

# Metric saving function
def save_metrics(loader, dataset_name, model, dataset):
    preds, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images, lbls = images.to(DEVICE), lbls.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dataset_name}_confusion_matrix.png"))
    plt.close()

    report = classification_report(labels, preds, target_names=dataset.get_classes())
    with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_classification_report.txt"), "w") as f:
        f.write(report)

# Grid Search Configuration
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

# Run grid search
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

grid_combinations = list(itertools.product(*param_grid.values()))

# Main Training Loop
for i, params in enumerate(grid_combinations):
    (optimizer_fn, lr, dropout, freeze, batch_size, epochs, activation, 
     (weight1, weight2), augment_threshold, downsample_threshold) = params

    dataset = load_dataset(DATA_DIR, transform)
    class_counts = analyze_classes(dataset)
    augment_classes, downsample_classes = determine_sampling_classes(
        class_counts, augment_threshold, downsample_threshold
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    resnet_model = ResNet50Classifier(
        num_classes=len(dataset.get_classes()),
        dropout_rate=dropout,
        freeze_layers=freeze,
        activation=activation
    ).to(DEVICE)

    biomedclip_model = BiomedCLIPClassifier(
        num_classes=len(dataset.get_classes()),
        dropout_rate=dropout,
        activation=activation,
        freeze_layers=freeze
    )

    ensemble_model = EnsembleModel(
        resnet_model,
        biomedclip_model,
        weight1=weight1,
        weight2=weight2
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_fn(ensemble_model.parameters(), lr=lr)

    for epoch in range(epochs):
        ensemble_model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = ensemble_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        save_metrics(train_loader, "train", ensemble_model, dataset)
        save_metrics(val_loader, "validation", ensemble_model, dataset)

        torch.save(ensemble_model.state_dict(), os.path.join(OUTPUT_DIR, f"ensemble_model_{i+1}.pth"))
