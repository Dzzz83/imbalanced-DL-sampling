import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('./LAVA')
from models.preact_resnet import PreActResNet18

def train_feature_extractor(config, gpu=1):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = config.get('epochs', 200)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    model = PreActResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['wd'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if config.get('clip', True):
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
        print(f"[{config['name']}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        if torch.isnan(torch.tensor(avg_loss)):
            print(f"[{config['name']}] NaN detected – stopping early.")
            break

    out_file = f"models/{config['name']}.pth"
    torch.save(model.state_dict(), out_file)
    print(f"[{config['name']}] Saved to {out_file}\n")

if __name__ == "__main__":
    experiments = [
        {
            'name': 'exp1_baseline_lr0.01_wd1e-4_clipTrue_mil120',
            'lr': 0.01,
            'wd': 1e-4,
            'clip': True,
            'milestones': [120, 160],
            'epochs': 200
        },
        {
            'name': 'exp2_lr0.05_wd5e-4_clipTrue_mil120',
            'lr': 0.05,
            'wd': 5e-4,
            'clip': True,
            'milestones': [120, 160],
            'epochs': 200
        },
        {
            'name': 'exp3_lr0.05_wd1e-3_clipTrue_mil120',
            'lr': 0.05,
            'wd': 1e-3,
            'clip': True,
            'milestones': [120, 160],
            'epochs': 200
        },
        {
            'name': 'exp4_lr0.02_wd5e-4_clipTrue_mil120',
            'lr': 0.02,
            'wd': 5e-4,
            'clip': True,
            'milestones': [120, 160],
            'epochs': 200
        },
        {
            'name': 'exp5_lr0.05_wd5e-4_clipTrue_mil100_150',
            'lr': 0.05,
            'wd': 5e-4,
            'clip': True,
            'milestones': [100, 150],
            'epochs': 200
        },
        {
            'name': 'exp6_lr0.01_wd5e-4_clipFalse_mil120',
            'lr': 0.01,
            'wd': 5e-4,
            'clip': False,
            'milestones': [120, 160],
            'epochs': 200
        },
    ]

    for config in experiments:
        print(f"\n=== Starting experiment: {config['name']} ===")
        train_feature_extractor(config, gpu=1)   # use GPU 1