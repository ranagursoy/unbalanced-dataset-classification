import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model_dataset import BalancedSpermDataset, ResNet50Classifier, EnsembleModel, BiomedCLIPClassifier
import timm
import os

# create_model tanımı
def create_model(model_name, pretrained=False):
    return timm.create_model(model_name, pretrained=pretrained)

# BiomedCLIPClassifier tanımı
class BiomedCLIPClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, activation=torch.nn.ReLU):
        super(BiomedCLIPClassifier, self).__init__()
        self.base_model = create_model("vit_base_patch16_224", pretrained=False)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.base_model.head.in_features, 512),
            activation(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model.forward_features(x)
        pooled_features = features[:, 0]
        return self.fc(pooled_features)

# Test veri seti
test_data_dir = "test"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = BalancedSpermDataset(test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yükleme
resnet_model = ResNet50Classifier(num_classes=len(test_dataset.get_classes()))
biomedclip_model = BiomedCLIPClassifier(num_classes=len(test_dataset.get_classes()))
ensemble_model = EnsembleModel(resnet_model, biomedclip_model).to(device)

# Kaydedilen ağırlıkları yükle
model_path = "grid_search_results_new2/ensemble_model_11.pth" 
ensemble_model.load_state_dict(torch.load(model_path))
ensemble_model.eval()

# Test seti üzerinde tahmin
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = ensemble_model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion Matrix ve Classification Report
report = classification_report(all_labels, all_preds, target_names=test_dataset.get_classes())
print(report)
with open("test_classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.get_classes(), yticklabels=test_dataset.get_classes())
plt.title('Test Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("test_confusion_matrix.png")
plt.show()
