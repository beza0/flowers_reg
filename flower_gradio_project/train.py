import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from model import FlowerCNN

# -----------------------------
# AYARLAR
# -----------------------------
DATA_DIR = "data/flowers"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# -----------------------------
# VERİ DÖNÜŞÜMLERİ
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -----------------------------
# DATASET ve DATALOADER
# -----------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

print("Sınıflar:", class_names)

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# MODEL
# -----------------------------
model = FlowerCNN(num_classes=len(class_names))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.model.fc.parameters(), lr=LEARNING_RATE)

# -----------------------------
# EĞİTİM
# -----------------------------
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss:.4f} | Accuracy: {accuracy:.2f}%")

# -----------------------------
# MODELİ KAYDET
# -----------------------------
torch.save(model.state_dict(), "flowers_model.pth")
print("✅ Model kaydedildi: flowers_model.pth")
