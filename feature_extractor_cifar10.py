# train_cifar10_feature_extractor.py
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Use the test set (10,000 images)
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    model = PreActResNet18()
    model = model.to(device)

    # Lower learning rate to prevent explosion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Check inputs (first batch only)
            if i == 0:
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"NaN/Inf in inputs at epoch {epoch}")
                    break
                if inputs.min() < -5 or inputs.max() > 5:
                    print(f"Input range: min={inputs.min():.3f}, max={inputs.max():.3f}")

            optimizer.zero_grad()
            outputs = model(inputs)

            # Check outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"NaN/Inf in outputs at epoch {epoch}, batch {i}")
                print(f"Output stats: min={outputs.min():.3f}, max={outputs.max():.3f}, mean={outputs.mean():.3f}")
                # Save model state for inspection
                torch.save(model.state_dict(), 'debug_model_nan.pth')
                return

            loss = criterion(outputs, labels)

            # Check loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at epoch {epoch}, batch {i}")
                return

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        # Early stopping if loss becomes NaN
        if torch.isnan(torch.tensor(avg_loss)):
            print("NaN detected in average loss, stopping.")
            break

    torch.save(model.state_dict(), 'models/cifar10_embedder_preact_resnet18_2.pth')
    print("Feature extractor saved to models/cifar10_embedder_preact_resnet18_2.pth")

if __name__ == "__main__":
    train_feature_extractor()