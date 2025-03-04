from random import random

import kiwisolver
import torch
import torch.nn.functional as F
from torch import autograd, nn
from loss.selective_loss import Loss_DC


class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            if grad_outputs.dtype == torch.float16:
                grad_outputs = grad_outputs.to(torch.float32)
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0


    def ema(self,x,decay):
        lut = self.lut*decay + x*(1-decay)
        self.lut = lut

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0

        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # input 只有前景
        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        # projected - Tensor [M, lut+cq], e.g., [M, 482+500]=[M, 982]

        projected *= self.oim_scalar

        self.header_cq = (self.header_cq + (label >= self.num_pids).long().sum().item()
                          ) % self.num_unlabeled

        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        return loss_oim, inputs, label


class LOIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum, ious, eps):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum, ious, eps)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum, ious, eps = ctx.saved_tensors
        ious = torch.clamp(ious, max=1 - eps)

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            if grad_outputs.dtype == torch.float16:
                grad_outputs = grad_outputs.to(torch.float32)
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y, s in zip(inputs, targets, ious.view(-1)): # target
            if y < len(lut):
                lut[y] = (1.0 - s) * lut[y] + s * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None, None, None


def loim(inputs, targets, lut, cq, header, momentum=0.5, ious=1.0, eps=0.2):
    return LOIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum), ious, torch.tensor(eps))



class newOIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum, ious, eps):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum, ious, eps)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum, ious, cls, eps = ctx.saved_tensors
        ious = torch.clamp(ious, max=1 - eps)

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y, s in zip(inputs, targets, ious.view(-1)): # target
            if y < len(lut):
                lut[y] = (1.0 - s) * lut[y] + s * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None, None, None

class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0


    def ema(self,x,decay):
        lut = self.lut*decay + x*(1-decay)
        self.lut = lut

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        # print(f'label:{label.shape}---{label}')
        inds = label >= 0

        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # input 只有前景
        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        # projected - Tensor [M, lut+cq], e.g., [M, 482+500]=[M, 982]

        projected *= self.oim_scalar

        self.header_cq = (self.header_cq + (label >= self.num_pids).long().sum().item()
                          ) % self.num_unlabeled
        # print(f'projected:{projected.shape}---{projected}')
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        return loss_oim, inputs, label

def newoim(inputs, targets, lut, cq, header, momentum=0.5, ious=1.0, eps=0.2):
    return newOIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum), ious, torch.tensor(eps))

class newOIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, eps):
        super(newOIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.oim_eps = eps

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0
        self.corr_loss = Loss_DC()
        self.focal_loss = FocalLoss(ignore_index=5554)

    def forward(self, inputs, roi_label, ious, cls):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        corr = self.corr_loss(target=inputs.clone(),
                              ious=ious.clone(),
                              cls=cls.clone(),
                              label=label)
        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, self.lut, self.cq, self.header_cq,momentum=self.momentum)

        projected *= self.oim_scalar

        self.header_cq = (self.header_cq + (label >= self.num_pids).long().sum().item()
                          ) % self.num_unlabeled

        # loss_oim = F.cross_entropy(projected, label, ignore_index=5554)+corr*1e-5
        loss_oim = self.focal_loss(projected, label) + 1e-3* corr
        return loss_oim, inputs, label


class LOIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, eps):
        super(LOIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.oim_eps = eps

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features, dtype=torch.float16))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features, dtype=torch.float16))

        self.header_cq = 0

        # self.bottleneck = nn.BatchNorm1d(self.num_features)
        # self.bottleneck.bias.requires_grad_(False)  # no shift
        # self.classifier = nn.Linear(self.num_features, self.num_pids, bias=False)
        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        # #  待改
        self.focal_loss =FocalLoss(ignore_index=5554)


    def forward(self, inputs, roi_label, ious):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        inds = label >= 0
        label = label[inds]
        ious = ious[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # tri_loss
        # projected, labeled_matching_reid, labeled_matching_ids = loim(inputs, label, self.lut, self.cq, self.header_cq,
        #                                                             momentum=self.momentum, ious=ious,eps=self.oim_eps)
        projected = loim(inputs, label, self.lut, self.cq, self.header_cq,
                         momentum=self.momentum, ious=ious,
                         eps=self.oim_eps)
        projected *= self.oim_scalar
        self.header_cq = (self.header_cq + (label >= self.num_pids).long().sum().item()
                          ) % self.num_unlabeled
        # loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        # return loss_oim, inputs, label
        loss_oim = self.focal_loss(projected, label)
        # pos_reid = torch.cat((inputs, labeled_matching_reid), dim=0)
        # pid_labels = torch.cat((label, labeled_matching_ids), dim=0)
        # loss_tri = 1e-8*self.tri_loss(pos_reid, pid_labels)
        # loss_tri = self.tri_loss(pos_reid, pid_labels)
        # print(label.shape)
        # inds2 = label != 5554
        # label2 = label[inds2]
        # ious2 = ious[inds2].detach()

        # target2 = label.clone()
        # #score = self.classifier(self.bottleneck(inputs))
        # # loss_softmax = F.cross_entropy(score,target2,ignore_index=5554)
        # score = projected[inds2,label2]
        # loss_corr = 0.1 #self.corr_loss(score.sigmoid(), ious2)
        return loss_oim,inputs, label


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss
