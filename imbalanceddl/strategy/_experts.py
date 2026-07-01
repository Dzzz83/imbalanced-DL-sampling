import torch
import torch.optim as optim
import math
from ..loss import LogitAdjustedLoss, BalancedSoftmaxLoss

class ExpertsTrainer:
    def __init__(self, cfg, model, train_loader, val_loader, cls_num_list):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.criterion_la = LogitAdjustedLoss(cls_num_list, tau=1.0).to(self.device)
        self.criterion_bs = BalancedSoftmaxLoss(cls_num_list).to(self.device)
        self.losses = [self.criterion_ce, self.criterion_la, self.criterion_bs]

        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=cfg.lr, 
            momentum=cfg.momentum, 
            weight_decay=cfg.weight_decay
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            experts_logits = self.model(images)
            
            loss = 0.0
            for i, logits in enumerate(experts_logits):
                loss += self.losses[i](logits, targets)
            
            loss /= len(experts_logits)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            # Compute accuracy using averaged probabilities across experts
            probs = [torch.softmax(logits, dim=1) for logits in experts_logits]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            _, predicted = avg_probs.max(1)
            total_samples += targets.size(0)
            total_correct += predicted.eq(targets).sum().item()
            
        return total_loss / total_samples, 100.0 * total_correct / total_samples

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        for images, targets, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            experts_logits = self.model(images)
            probs = [torch.softmax(logits, dim=1) for logits in experts_logits]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            
            _, predicted = avg_probs.max(1)
            total_samples += targets.size(0)
            total_correct += predicted.eq(targets).sum().item()
            
        accuracy = 100.0 * total_correct / total_samples
        return accuracy

    def do_train(self):
        best_acc = 0.0
        for epoch in range(self.cfg.epochs):
            self.adjust_learning_rate(epoch)
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_acc = self.validate()
            
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc)

    def adjust_learning_rate(self, epoch):
        # Cosine Annealing LR schedule
        lr = self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.cfg.epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, epoch, acc):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_acc': acc,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, f"{self.cfg.root_model}/checkpoint_experts_epoch{epoch}.pth")