import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average="mean", ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        '''
        :param predict: (n, c, h, w)
        :param target: (n,[c=1], h, w)
        :param weight: (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        :return: Loss, refer to deepblue
        '''
        target = target.squeeze()
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].to(torch.long)
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.size_average)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()

        """
        Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
            ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, p, target):
        """
        :param input: (n, c, h, w)
        :param target: (n, c, h, w)
        :return: focal loss
        """
        p = torch.sigmoid(p)
        mask = (target == 0)
        pos_p = target.type(torch.float) - p
        pos_p[mask] = 0
        neg_p = p.new_tensor(p.data)
        neg_p[~mask] = 0

        loss = -1 * self.alpha * pos_p ** self.gamma * (p + 1e-8).log() \
               - (1 - self.alpha) * neg_p ** self.gamma * (1 - p + 1e-8).log()

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
