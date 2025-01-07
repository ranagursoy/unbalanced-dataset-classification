import os
import itertools
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model_dataset import BalancedSpermDataset, ResNet50Classifier, VGG16Classifier, EnsembleModel

# Çıkış klasörü
output_dir = "grid_search_results"
os.makedirs(output_dir, exist_ok=True)

# Veri seti yolu
data_dir = "train"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sınıfları analiz etmek için fonksiyon
def analyze_classes(dataset):
    """
    Veri setindeki her sınıfın örnek sayısını analiz eder.
    """
    class_counts = {class_name: 0 for class_name in dataset.get_classes()}
    for _, label in dataset:
        class_counts[dataset.get_classes()[label]] += 1
    return class_counts

# Dinamik olarak augment ve downsample sınıflarını belirleme
def determine_sampling_classes(class_counts, augment_threshold=500, downsample_threshold=1000):
    """
    Sınıf örnek sayılarına göre augment ve downsample sınıflarını belirler.
    """
    augment_classes = [i for i, count in enumerate(class_counts.values()) if count < augment_threshold]
    downsample_classes = [i for i, count in enumerate(class_counts.values()) if count > downsample_threshold]
    return augment_classes, downsample_classes

# Veri seti ve sınıf analizi
dataset = BalancedSpermDataset(data_dir, transform=transform)
class_counts = analyze_classes(dataset)
print("Sınıf Analizi:", class_counts)

# Parametre kombinasyonları
param_grid = {
    "optimizer": [torch.optim.Adam, torch.optim.SGD],
    "learning_rate": [0.001],
    "dropout_rate": [0.2],
    "freeze_layers": [0, 3, 5],
    "batch_size": [16, 32],
    "epochs": [20, 50],
    "activation": [torch.nn.ReLU, torch.nn.LeakyReLU],
    "weights": [(0.5, 0.5), (0.7, 0.3)],
    "augment_threshold": [300, 500],
    "downsample_threshold": [800, 1000]
}
grid_combinations = list(itertools.product(*param_grid.values()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_index = 30

for i, params in enumerate(grid_combinations):
    if i < start_index:
        continue
    (optimizer_fn, lr, dropout, freeze, batch_size, epochs, activation, 
     (weight1, weight2), augment_threshold, downsample_threshold) = params

    print(f"Configuration {i+1}/{len(grid_combinations)}: {params}")

    # Dinamik augment ve downsample sınıfları belirleme
    augment_classes, downsample_classes = determine_sampling_classes(
        class_counts,
        augment_threshold=augment_threshold,
        downsample_threshold=downsample_threshold
    )

    # Dengeli veri seti oluşturma
    dataset = BalancedSpermDataset(
        data_dir=data_dir,
        transform=transform,
        augment_classes=augment_classes,
        downsample_classes=downsample_classes,
        max_samples_per_class=downsample_threshold,
        upsample_factor=2
    )

    # Train/Validation/Test bölünmesi
    test_size = int(0.2 * len(dataset))
    train_val_size = len(dataset) - test_size
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

    val_size = int(0.2 * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Ensemble model
    resnet_model = ResNet50Classifier(
        num_classes=len(dataset.get_classes()),
        dropout_rate=dropout,
        freeze_layers=freeze,
        activation=activation
    ).to(device)

    vgg16_model = VGG16Classifier(
        num_classes=len(dataset.get_classes()),
        dropout_rate=dropout,
        activation=activation
    ).to(device)

    ensemble_model = EnsembleModel(
        resnet_model,
        vgg16_model,
        weight1=weight1,
        weight2=weight2
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_fn(ensemble_model.parameters(), lr=lr)

    train_losses, val_losses = [], []
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

        ensemble_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = ensemble_model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

    # Sonuçları kaydetme
    model_folder = os.path.join(output_dir, f"model_{i+1}")
    os.makedirs(model_folder, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(model_folder, "loss_curve.png"))
    plt.close()

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
        plt.savefig(os.path.join(model_folder, f"{dataset_name}_confusion_matrix.png"))
        plt.close()

        report = classification_report(labels, preds, target_names=dataset.get_classes())
        with open(os.path.join(model_folder, f"{dataset_name}_classification_report.txt"), "w") as f:
            f.write(report)

    save_metrics(train_loader, "train")
    save_metrics(val_loader, "validation")
    save_metrics(test_loader, "test")