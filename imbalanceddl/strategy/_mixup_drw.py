import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer

from imbalanceddl.utils.utils import AverageMeter
from imbalanceddl.utils.metrics import accuracy
import wandb.apis.public as public
import wandb
from imbalanceddl.utils.m2m_utils import InputNormalize
import torch.nn.functional as F


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MixupTrainer(Trainer):
    """Mixup-DRW Trainer

    Strategy: Mixup with DRW training schedule

    Here we provide Mixup-DRW as a strategy, if you want to test
    original Mixup on imbalanced dataset, just change criterion
    in get_criterion() method.

    Reference
    ----------
    Paper: mixup: Beyond Empirical Risk Minimization
    Paper Link: https://arxiv.org/pdf/1710.09412.pdf
    Code: https://github.com/facebookresearch/mixup-cifar10

    Paper (DRW): Learning Imbalanced Datasets with \
    Label-Distribution-Aware Margin Loss
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_criterion(self):
        if self.strategy == 'Mixup_DRW':
            if self.cfg.epochs == 300:
                idx = self.epoch // 250
            else:
                idx = self.epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(
                self.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(
                self.cfg.gpu)
            print("=> Per Class Weight = {}".format(per_cls_weights))
            self.criterion = nn.CrossEntropyLoss(weight=per_cls_weights,
                                                 reduction='none').cuda(
                                                     self.cfg.gpu)
        else:
            raise ValueError("[Warning] Strategy is not supported !")
    def train_one_epoch_balance_mixup(self):
        
        if self.cfg.dataset == 'cifar100':
            mean = torch.tensor([0.5071, 0.4867, 0.4408])
            std = torch.tensor([0.2675, 0.2565, 0.2761])
        elif self.cfg.dataset == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
        elif self.cfg.dataset == 'svhn10':
            mean = torch.tensor([.5, .5, .5])
            std = torch.tensor([.5, .5, .5])
        elif self.cfg.dataset == 'tiny200':
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
        elif self.cfg.dataset == 'cinic10':
            mean = torch.tensor([0.47889522, 0.47227842, 0.43047404])
            std = torch.tensor([0.24205776, 0.23828046, 0.25874835])
        else:
            raise NotImplementedError()

        normalizer = InputNormalize(mean, std).to(self.cfg.gpu)

        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # for confusion matrix
        all_preds = list()
        all_targets = list()

        uncertainty_samples = list()

        # med_tail_classes = cal_med_tail_classes(self.new_labelList, self.cfg.num_classes)
        # med_tail_classes = torch.from_numpy(med_tail_classes).to(self.cfg.gpu)
        # switch to train mode
        self.model.to(self.cfg.gpu)
        self.model.train()

        epoch_ave_grads_train = []
        # epoch_ave_losses = []

        # for i, (_input, target) in enumerate(self.train_loader_custom):
        for i, (_input, target) in enumerate(self.train_oversamples): 

            if self.cfg.gpu is not None:
                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

            _input = normalizer(_input)
            # print("=> Training with Original Mixup")
            # Mixup Data
            _input_mix, target_a, target_b, lam = mixup_data(_input, target)
            # Two kinds of output
            output_prec, _ = self.model(_input)
            output_mix, _ = self.model(_input_mix)
            # output_prec = self.model(_input)
            # output_mix = self.model(_input_mix)

            # Probability of output
            # prob_orgi = F.softmax(output_prec, dim=1)
            prob_mix = F.softmax(output_mix, dim=1)
            # max_prob_orgi, target_orig = torch.max(prob_orgi, dim=1)
            max_prob_mix, target_mix = torch.max(prob_mix, dim=1)

            # Calculate mask/ self.cfg.threshold is a hyper parameter to fine tunning uncertainty samples
            # mask_orig = max_prob_orgi.le(self.cfg.threshold).float()
            # mask_mix = max_prob_mix.le(self.cfg.threshold).float() #* check_in_med_tail(target_mix, med_tail_classes)

            # Count the number of uncertainty samples
            target_mix = target_mix.cpu().numpy() #* mask_mix.cpu().numpy()
            uncertainty_samples.append(target_mix)

            # loss_orgi = (mask_orig * self.criterion(output_prec, target)).mean()
            # loss_mix = (mask_mix * mixup_criterion(self.criterion, output_mix, target_a,target_b, lam)).mean()
            loss_mix = mixup_criterion(self.criterion, output_mix, target_a,target_b, lam).mean()

            # final_loss = loss_orgi + loss_mix
            final_loss = loss_mix
            # For Loss, we use mixup output
            # loss = mixup_criterion(self.criterion, output_mix, target_a,
            #                        target_b, lam).mean()
            acc1, acc5 = accuracy(output_prec, target, topk=(1, 5))
            _, pred = torch.max(output_prec, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure accuracy and record loss
            # losses.update(loss.item(), _input.size(0))
            losses.update(final_loss.item(), _input.size(0))
            top1.update(acc1[0], _input.size(0))
            top5.update(acc5[0], _input.size(0))

            self.optimizer.zero_grad()
            # loss.backward()
            final_loss.backward()
            self.optimizer.step()

            # Checking gradient vector
            # model = self.model.cpu().named_parameters()
            # ave_grads, self.layers = plot_grad_flow(model)
            # ave_grads = np.asarray(ave_grads)
            # epoch_ave_grads_train.append(ave_grads)
            # self.model.to(self.cfg.gpu)

            if i % self.cfg.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              self.epoch,
                              i,
                              len(self.train_loader),
                              loss=losses,
                              top1=top1,
                              top5=top5,
                              lr=self.optimizer.param_groups[-1]['lr'] * 0.1))
                print(output)
                self.log_training.write(output + '\n')
                self.log_training.flush()

        # epoch_ave_grads_train = np.asarray(epoch_ave_grads_train)
        # epoch_ave_grads_train = np.mean(epoch_ave_grads_train, axis = 0)
        # if self.epoch == 165:
        #     self.epoch_ave_grads_165 = epoch_ave_grads_train
        # elif self.epoch == 170:
        #     self.epoch_ave_grads_170 = epoch_ave_grads_train
        # elif self.epoch == self.cfg.epochs - 1:
        #     self.epoch_ave_grads_200 = epoch_ave_grads_train
        
        # self.epoch_variance_train = np.sum(epoch_ave_grads_train)

        # epoch_ave_losses = np.asanyarray(losses.avg)
        # self.trn_epoch_accuracy_avg = np.sum(epoch_ave_losses)

        # Count the number of uncertainty samples for each class
        uncertainty_samples = np.concatenate(uncertainty_samples, axis=0)
        classes, class_counts = np.unique(uncertainty_samples, return_counts=True)
        # import pdb
        # pdb.set_trace()
        # print("The classes: {}".format(classes))
        print("The number of samples in each class of Balanced + Mixup: {}".format(class_counts))
        self.compute_metrics_and_record(all_preds,
                                        all_targets,
                                        losses,
                                        top1,
                                        top5,
                                        flag='Training')
        
        self.trn_epoch_accuracy_avg = top1.avg
    def train_one_epoch(self):
        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # for confusion matrix
        all_preds = list()
        all_targets = list()

        # switch to train mode
        self.model.train()

        for i, (_input, target) in enumerate(self.train_loader):

            if self.cfg.gpu is not None:
                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

            # print("=> Training with Original Mixup")
            # Mixup Data
            _input_mix, target_a, target_b, lam = mixup_data(_input, target)
            # Two kinds of output
            output_prec, _ = self.model(_input)
            output_mix, _ = self.model(_input_mix)
            # For Loss, we use mixup output
            loss = mixup_criterion(self.criterion, output_mix, target_a,
                                   target_b, lam).mean()
            acc1, acc5 = accuracy(output_prec, target, topk=(1, 5))
            _, pred = torch.max(output_prec, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure accuracy and record loss
            losses.update(loss.item(), _input.size(0))
            top1.update(acc1[0], _input.size(0))
            top5.update(acc5[0], _input.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.cfg.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              self.epoch,
                              i,
                              len(self.train_loader),
                              loss=losses,
                              top1=top1,
                              top5=top5,
                              lr=self.optimizer.param_groups[-1]['lr'] * 0.1))
                print(output)
                self.log_training.write(output + '\n')
                self.log_training.flush()
        wandb.log({
        "epoch": self.epoch,
        "epoch_train_loss": losses.avg,
        "epoch_train_acc@1": top1.avg,
        "epoch_train_acc@5": top5.avg,
        "lr": self.optimizer.param_groups[-1]['lr'] * 0.1
        })

        self.compute_metrics_and_record(all_preds,
                                        all_targets,
                                        losses,
                                        top1,
                                        top5,
                                        flag='Training')
