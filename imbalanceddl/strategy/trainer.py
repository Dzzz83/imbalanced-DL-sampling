import os
import torch
import torch.optim as optim
import numpy as np
from imbalanceddl.utils.utils import AverageMeter, save_checkpoint, collect_result
from imbalanceddl.utils.metrics import accuracy
from .base import BaseTrainer
from imbalanceddl.utils.m2m_utils import Logger
from torchmetrics import F1Score
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score
import wandb
import wandb.apis.public as public
from torch.utils.data import DataLoader

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyward-only argument: 'model' !")
        else:
            print("=> Model = {}".format(self.model))
        self.strategy = kwargs.pop('strategy', None)
        if self.strategy is None:
            raise TypeError("__init__() missing required keyward-only \
                argument: 'strategy' !")
        else:
            print("=> Strategy = {}".format(self.strategy))
        self.optimizer = self._init_optimizer()
        self.cls_num_list = self.cfg.cls_num_list
        self.img_num_per_cls = self.cfg.cls_num_list
        self.best_acc1 = 0.
        self.best_val_acc1 = 0.

        if self.cfg.strategy == "M2m":
            LOGFILE_BASE = f"S{self.cfg.seed}_{self.cfg.strategy}_" \
                f"L{self.cfg.lam}_W{self.cfg.warm}_" \
                f"E{self.cfg.step_size}_I{self.cfg.attack_iter}_" \
                f"{self.cfg.dataset}_R{int(1/self.cfg.imb_factor)}_{self.cfg.backbone}_G{self.cfg.gamma}_B{self.cfg.beta}"
                
            LOGNAME = 'Imbalance_' + LOGFILE_BASE
            self.logger = Logger(LOGNAME)
            self.LOGDIR = self.logger.logdir
            self.LOG_CSV = os.path.join(self.LOGDIR, f'log_{self.cfg.seed}.csv')
            self.LOG_CSV_HEADER = [
                'epoch', 'train loss', 'gen loss', 'train acc', 'gen_acc', 'prob_orig', 'prob_targ',
                'test loss', 'major test acc', 'neutral test acc', 'minor test acc', 'test acc', 'f1 score'
            ]

        # Conditional WandB initialisation
        self.use_wandb = getattr(self.cfg, 'use_wandb', False)
        if self.use_wandb:
            wandb.login(key="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")
            wandb.init(
                project="imbalanced_training",
                config={
                    "dataset": self.cfg.dataset,
                    "imb_type": self.cfg.imb_type,
                    "imb_factor": self.cfg.imb_factor,
                    "backbone": self.cfg.backbone,
                    "classifier": self.cfg.classifier,
                    "optimizer": self.cfg.optimizer,
                    "learning_rate": self.cfg.learning_rate,
                    "momentum": self.cfg.momentum,
                    "weight_decay": self.cfg.weight_decay,
                    "batch_size": self.cfg.batch_size,
                    "epochs": self.cfg.epochs,
                    "sampling_method": self.cfg.sampling,
                    "alpha": self.cfg.alpha,
                    "n_batches": self.cfg.n_batches,
                    "selection_method": self.cfg.selection_method
                },
                name=(
                    f"{self.cfg.dataset}_{self.cfg.backbone}_"
                    f"Select-{self.cfg.selection_method}_" 
                    f"Sample-{self.cfg.sampling}" 
                    + (f"_alpha{self.cfg.alpha}" if self.cfg.sampling in ["WeightedRandomBatchSampler", "WeightedFixedBatchSampler"] else "")
                )
            )

    def get_criterion(self):
        return NotImplemented

    def train_one_epoch(self):
        return NotImplemented

    def _init_optimizer(self):
        if self.cfg.optimizer == 'sgd':
            print("=> Initialize optimizer {}".format(self.cfg.optimizer))
            optimizer = optim.SGD(self.model.parameters(),
                                  self.cfg.learning_rate,
                                  momentum=self.cfg.momentum,
                                  weight_decay=self.cfg.weight_decay)
            return optimizer
        elif self.cfg.optimizer == 'adam':
            print("=> Initialize optimizer {}".format(self.cfg.optimizer))
            optimizer = optim.Adam(self.model.parameters(),
                                   self.cfg.learning_rate,
                                   weight_decay = self.cfg.weight_decay)
            return optimizer
        else:
            raise ValueError("[Warning] Selected Optimizer not supported !")

    def adjust_learning_rate(self):
        """Sets the learning rate"""
        if self.cfg.epochs == 200:
            epoch = self.epoch + 1
            if epoch <= 5:
                lr = self.cfg.learning_rate * epoch / 5
            elif epoch > 180:
                lr = self.cfg.learning_rate * 0.0001
            elif epoch > 160:
                lr = self.cfg.learning_rate * 0.01
            else:
                lr = self.cfg.learning_rate
        elif self.cfg.epochs == 300:
            epoch = self.epoch + 1
            if epoch <= 5:
                lr = self.cfg.learning_rate * epoch / 5
            elif epoch > 250:
                lr = self.cfg.learning_rate * 0.01
            elif epoch > 150:
                lr = self.cfg.learning_rate * 0.1
            else:
                lr = self.cfg.learning_rate
        elif self.cfg.epochs == 400:
            epoch = self.epoch + 1
            if epoch <=5:
                lr = self.cfg.learning_rate * epoch / 5
            elif epoch > 320:
                lr = self.cfg.learning_rate * 0.01
            elif epoch > 250:
                lr = self.cfg.learning_rate * 0.1
            else:
                lr = self.cfg.learning_rate
        else:
            raise ValueError(
                "[Warning] Total epochs {} not supported !".format(
                    self.cfg.epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def do_train_val_m2m(self):
        device = self.cfg.gpu
        self.logger.log('==> Building model: %s' % self.cfg.backbone)
        net = self.model
        net_seed = self.model
        net, net_seed = net.to(device), net_seed.to(device)
        optimizer = self._init_optimizer()
        SUCCESS = torch.zeros(self.cfg.epochs, self.cfg.num_classes, 2)
        self.train_oversamples = []

        if self.cfg.over:
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                self.epoch = epoch
                self.train_one_epoch(net, net_seed, optimizer, SUCCESS)

    def do_train_val(self):
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.epoch = epoch
            self.adjust_learning_rate()
            self.get_criterion()
            assert self.criterion is not None, "No criterion !"
            self.train_one_epoch()
            acc1 = self.validate()
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            output_best = f'Best Prec@1: {self.best_acc1:.3f}\n'
            print(output_best)
            if self.log_testing is not None:
                self.log_testing.write(output_best)
                self.log_testing.flush()

            save_checkpoint(
                self.cfg, {
                    'epoch': self.epoch + 1,
                    'backbone': self.cfg.backbone,
                    'classifier': self.cfg.classifier,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer': self.optimizer.state_dict()
                }, is_best, self.epoch)

    def eval_best_model(self):
        assert self.cfg.best_model is not None, "[Warning] Best Model must be loaded !"
        assert 'best' in self.cfg.best_model, "[Need Best Model]"

        if os.path.isfile(self.cfg.best_model):
            print("=> [Loading Best Model] '{}'".format(self.cfg.best_model))
            checkpoint = torch.load(self.cfg.best_model, map_location='cuda:0')
            self.epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if self.cfg.gpu is not None:
                best_acc1 = best_acc1.to(self.cfg.gpu)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> [Loaded Best Model] '{}' (epoch {})".format(
                self.cfg.best_model, checkpoint['epoch']))
        else:
            print("=> [No Trained Model Path found at '{}'".format(
                self.cfg.best_model))
            raise ValueError("[Warning] No Trained Model Path Found !!!")

        self.get_criterion()
        assert self.criterion is not None, "No criterion !"
        acc1, cls_acc_string = self.validate()
        output_best = 'Best Prec@1: %.3f' % (acc1)
        print(output_best)
        print(cls_acc_string)
        print("[Done] with evaluating with best model of {}".format(
            self.cfg.best_model))
        return

    def validate(self):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                if self.cfg.gpu is not None:
                    images = images.cuda(self.cfg.gpu, non_blocking=True)
                    labels = labels.cuda(self.cfg.gpu, non_blocking=True)

                output, _ = self.model(images)
                labels = labels.to(output.device)
                loss = self.criterion(output, labels.to(output.device)).mean()
                acc1, acc5 = accuracy(output, labels.to(output.device), topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                if i % self.cfg.print_freq == 0:
                    stats_log = ('Epoch: [{0}][{1}/{2}]\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                     self.epoch,
                                     i,
                                     len(self.val_loader),
                                     loss=losses,
                                     top1=top1,
                                     top5=top5))
                    print(stats_log)

            all_preds_t = torch.tensor(all_preds)
            all_targets_t = torch.tensor(all_targets)
            if self.cfg.gpu is not None:
                all_preds_t = all_preds_t.cuda(self.cfg.gpu)
                all_targets_t = all_targets_t.cuda(self.cfg.gpu)

            final_f1 = multiclass_f1_score(all_preds_t, all_targets_t, num_classes=self.cfg.num_classes, average='macro').item()
            final_prec = multiclass_precision(all_preds_t, all_targets_t, num_classes=self.cfg.num_classes, average='macro').item()
            final_recall = multiclass_recall(all_preds_t, all_targets_t, num_classes=self.cfg.num_classes, average='macro').item()

            if self.use_wandb:
                wandb.log({
                    "val_loss": losses.avg,
                    "val_accuracy": top1.avg,
                    "val_f1_score": final_f1,
                    "val_precision": final_prec,
                    "val_recall": final_recall,
                    "epoch": self.epoch,
                })

            cls_acc_string = self.compute_metrics_and_record(all_preds,
                                                             all_targets,
                                                             losses,
                                                             top1,
                                                             top5,
                                                             flag='Testing')
            print("==========F1_Score of TESTING dataset: {:.4f}% =============".format(final_f1*100))

        if cls_acc_string is not None:
            return top1.avg, cls_acc_string
        else:
            return top1.avg