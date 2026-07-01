import torch
import torch.optim as optim
import math
from .base import BaseTrainer
from ..loss import LogitAdjustedLoss, BalancedSoftmaxLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ExpertsTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        super(ExpertsTrainer, self).__init__(cfg, dataset, **kwargs)
        self.model = kwargs.get('model')
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # cls_num_list is populated in cfg by ImbalancedDataset
        self.cls_num_list = cfg.cls_num_list
        
        # Define the 3 long-tail-aware losses
        self.criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion_la = LogitAdjustedLoss(self.cls_num_list, tau=1.0).to(self.device)
        self.criterion_bs = BalancedSoftmaxLoss(self.cls_num_list).to(self.device)
        self.losses = [self.criterion_ce, self.criterion_la, self.criterion_bs]

        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=cfg.lr, 
            momentum=cfg.momentum, 
            weight_decay=cfg.weight_decay
        )
        
        self.best_acc = 0.0

    def get_criterion(self):
        # Satisfies abstract method; actual losses handled in train_one_epoch
        return self.criterion_ce

    def adjust_learning_rate(self, epoch):
        # Cosine Annealing LR schedule
        lr = self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.cfg.epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Network returns out (list of 3 logits), hidden
            out, _ = self.model(images)
            experts_logits = out
            
            loss = 0.0
            for i, logits in enumerate(experts_logits):
                loss += self.losses[i](logits, targets)
            
            loss /= len(experts_logits)
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), images.size(0))
            
            # Compute accuracy using averaged probabilities across experts
            probs = [torch.softmax(logits, dim=1) for logits in experts_logits]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            _, predicted = avg_probs.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))
            
        return losses, top1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        top1 = AverageMeter()
        
        for images, targets, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            out, _ = self.model(images)
            experts_logits = out
            
            probs = [torch.softmax(logits, dim=1) for logits in experts_logits]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            
            _, predicted = avg_probs.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))
            
        return top1

    def do_train_val(self):
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            self.adjust_learning_rate(epoch)
            train_losses, train_top1 = self.train_one_epoch()
            val_top1 = self.validate()
            
            log_msg = f"Epoch {epoch}: Train Loss {train_losses.avg:.4f} | Train Acc {train_top1.avg:.2f}% | Val Acc {val_top1.avg:.2f}%"
            self.logger.info(log_msg)
            print(log_msg)
            
            if val_top1.avg > self.best_acc:
                self.best_acc = val_top1.avg
                self.save_checkpoint(epoch, val_top1.avg)

    def save_checkpoint(self, epoch, acc):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_acc': acc,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, f"{self.cfg.root_model}/checkpoint_experts_epoch{epoch}.pth")
        
    def eval_best_model(self):
        self.logger.info(f"=> Loading best model for evaluation: {self.cfg.best_model}")
        checkpoint = torch.load(self.cfg.best_model)
        self.model.load_state_dict(checkpoint['state_dict'])
        val_top1 = self.validate()
        eval_msg = f"=> Best Model Validation Accuracy: {val_top1.avg:.2f}%"
        self.logger.info(eval_msg)
        print(eval_msg)