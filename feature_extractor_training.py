# train_cifar100_feature_extractor.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('./LAVA')
from models.preact_resnet import PreActResNet18

def train_feature_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 200

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Load model with default 10 classes
    model = PreActResNet18()
    # Replace the final linear layer for 100 classes
    in_features = model.linear.in_features
    model.linear = nn.Linear(in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

    torch.save(model.state_dict(), 'models/cifar100_embedder_preact_resnet18.pth')
    print("Feature extractor saved to models/cifar100_embedder_preact_resnet18.pth")

if __name__ == "__main__":
    train_feature_extractor()