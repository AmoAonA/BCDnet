import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

class Loss_DC_(nn.Module):
    def __init__(self):
        super(Loss_DC_, self).__init__()

    def Distance_Correlation(self, latent, control):
        matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim=-1) + 1e-12)
        matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim=-1) + 1e-12)
        a_mean = torch.mean(matrix_a, dim=0, keepdim=True), torch.mean(matrix_a, dim=1, keepdim=True)
        b_mean = torch.mean(matrix_b, dim=0, keepdim=True), torch.mean(matrix_b, dim=1, keepdim=True)
        matrix_A = matrix_a - a_mean[0] - a_mean[1] + torch.mean(matrix_a)
        matrix_B = matrix_b - b_mean[0] - b_mean[1] + torch.mean(matrix_b)

        Gamma_XY = torch.sum(matrix_A * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = torch.sum(matrix_A * matrix_A) / (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = torch.sum(matrix_B * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
        correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r

    def forward(self, target, ious=None, cls=None, label=None):
        with torch.no_grad():
            label = torch.cat(label, dim=0)
            index = label > 0
            target = target[index.unsqueeze(1).expand_as(target)].view(-1, 256)
            ious = ious.unsqueeze(1)
            cls = cls.unsqueeze(1)
            ious = ious[index.unsqueeze(1).expand_as(ious)].view(-1, 1).detach()
            cls = cls[index.unsqueeze(1).expand_as(cls)].view(-1, 1).detach()
        dc_loss = 1 - self.Distance_Correlation(target, ious)
        dc_loss_ = self.Distance_Correlation(target, cls)
        return 0.5*dc_loss + dc_loss_
        # return dc_loss_
        # return dc_loss

def pearsonr(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    vx = x - x_mean
    vy = y - y_mean
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return 1 - pcc
# 1

class Loss_DC_base(nn.Module):
    def __init__(self):
        super(Loss_DC_base, self).__init__()

    def forward(self, target, ious=None, cls=None, label=None):
        with torch.no_grad():
            label = torch.cat(label, dim=0) # exp10 注释掉
            index = label > 0
            target = target[index.unsqueeze(1).expand_as(target)].view(-1, 256)
            ious = ious.unsqueeze(1)
            cls = cls.unsqueeze(1)
            ious = ious[index.unsqueeze(1).expand_as(ious)].view(-1, 1).detach()
            cls = cls[index.unsqueeze(1).expand_as(cls)].view(-1, 1).detach()
        dc_loss = 1 - Distance_Correlation(target, ious)
        dc_loss_ = Distance_Correlation(target, cls)
        return dc_loss + dc_loss_

class Loss_DC(nn.Module):
    def __init__(self):
        super(Loss_DC, self).__init__()

    def forward(self, target, ious=None, cls=None, label=None):
        with torch.no_grad():
            # label = torch.cat(label, dim=0)
            index = label > 0
            target = target[index.unsqueeze(1).expand_as(target)].view(-1, 256)
            ious = ious.unsqueeze(1)
            cls = cls.unsqueeze(1)
            ious = ious[index.unsqueeze(1).expand_as(ious)].view(-1, 1).detach()
            cls = cls[index.unsqueeze(1).expand_as(cls)].view(-1, 1).detach()
        dc_loss = 1 - Distance_Correlation(target, ious)
        dc_loss_ = Distance_Correlation(target, cls)
        #return dc_loss + dc_loss_
        return dc_loss_
        #return dc_loss


def Distance_Correlation(latent, control):
    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim=-1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim=-1) + 1e-12)
    a_mean = torch.mean(matrix_a, dim=0, keepdim=True), torch.mean(matrix_a, dim=1, keepdim=True)
    b_mean = torch.mean(matrix_b, dim=0, keepdim=True), torch.mean(matrix_b, dim=1, keepdim=True)
    matrix_A = matrix_a - a_mean[0] - a_mean[1] + torch.mean(matrix_a)
    matrix_B = matrix_b - b_mean[0] - b_mean[1] + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
    correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def test_loss(box_cls_scores, box_regs, box_labels, box_reg_targets, aa_regs, aa_clss):
    sigma = 0.5
    box_labels = torch.cat(box_labels, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)  # loss1+loss2
    cls_score_F = torch.squeeze(torch.bmm(aa_clss.unsqueeze(1), box_cls_scores.view(box_cls_scores.shape[0], 2, -1)))
    loss_rcnn_cls_a = sigma * F.cross_entropy(cls_score_F, box_labels)
    loss_rcnn_cls_s = (1 - sigma) * cross_entropy(box_cls_scores, box_labels, aa_clss)
    loss_rcnn_cls = loss_rcnn_cls_a + loss_rcnn_cls_s

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)
    box_regs_F = torch.bmm(aa_regs.unsqueeze(1), box_regs.view(N, 2, -1)).view(N, 2, 4)
    box_regs = box_regs.view(N, 2, 4 * 2)
    loss_rcnn_reg_a = sigma * F.smooth_l1_loss(box_regs_F[sampled_pos_inds_subset, labels_pos],
                                               box_reg_targets[sampled_pos_inds_subset], reduction="sum")
    loss_rcnn_cls_s = (1 - sigma) * smooth_l1_loss(box_regs[sampled_pos_inds_subset, labels_pos, :],
                                                   box_reg_targets[sampled_pos_inds_subset],
                                                   aa_regs[sampled_pos_inds_subset])
    loss_rcnn_reg = loss_rcnn_reg_a + loss_rcnn_cls_s
    loss_rcnn_reg = loss_rcnn_reg / box_labels.numel()
    return loss_rcnn_cls, loss_rcnn_reg


def cross_entropy(pred, label, weight=None, groups=1, reduction='mean', avg_factor=None, ):
    """Calculate the CrossEntropy loss with random node selection.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C * groups), C is the number
            of classes, groups is the number of nodes.
        aa (torch.Tensor): the soft routing probabilities
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    preds = torch.chunk(pred, 2, dim=1)
    bsz = pred.shape[0]
    num_nodes = 2
    num_trees = 1

    loss_ns = []
    for i, pred_i in enumerate(preds):
        loss_i = F.cross_entropy(pred_i, label, weight=None, reduction='none')
        loss_ns.append(loss_i.view(-1, 1))
    loss_ns = torch.cat(loss_ns, dim=1)
    loss_ns = loss_ns.view(loss_ns.shape[0], -1, num_nodes)

    target_n = loss_ns.argmin(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0])
    for i in range(num_trees):
        rho_min = np.random.uniform(0.1, 0.3)
        rho_max = np.random.uniform(0.9, 1.1)
        decay_[inds, i, :] = rho_min
        decay_[inds, i, target_n[inds, i]] = rho_max

    loss = loss_ns * decay_
    loss = loss.view(bsz, -1)
    loss = weight_reduce_loss(
        loss, weight=weight.unsqueeze(1), reduction=reduction, avg_factor=avg_factor)
    return 2 * loss / groups


def smooth_l1_loss(pred, target, aa, beta=1.0, weight=None, reduction='mean', avg_factor=None):
    """Smooth L1 loss for groups of predictions.
    Args:
        pred (torch.Tensor): The prediction.
        aa (torch.Tensor): the soft routing probabilities
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0

    assert pred.size(0) == target.size(0) and target.numel() > 0

    groups_ = 2
    preds = torch.chunk(pred, groups_, dim=1)
    batch_size = pred.shape[0]
    num_nodes = 2
    num_trees = int(groups_ / num_nodes)
    loss_ns = []
    for i, pred_i in enumerate(preds):
        diff = torch.abs(pred_i - target)
        loss_i = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        loss_i = torch.sum(loss_i, dim=1)
        loss_ns.append(loss_i.view(-1, 1))
    loss_ns = torch.cat(loss_ns, dim=1)
    loss_ns = loss_ns.view(batch_size, -1, num_nodes)
    aa = aa.view(batch_size, -1, num_nodes)

    target_n = aa.argmax(dim=2)
    # target_n = loss_ns.argmin(dim=2)
    decay_ = torch.ones_like(loss_ns)
    inds = range(loss_ns.shape[0])
    for i in range(num_trees):
        rho_min = np.random.uniform(0.1, 0.3)
        rho_max = np.random.uniform(0.9, 1.1)
        decay_[inds, i, :] = rho_min
        decay_[inds, i, target_n[inds, i]] = rho_max
    loss = loss_ns * decay_
    loss = loss.view(batch_size, -1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return 2. * loss / groups_
