import torch as t
from torch import nn


class CLSLoss(nn.Module):

    def __init__(self):
        super(CLSLoss, self).__init__()
        pass

    def forward(self, use_for_cls_loss, target):
        pass


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()
        pass

    def forward(self, point1, point2):
        """

        :param point1: registration result point, shape like (N, num_of_correspondence, 3)
        :param point2: ground truth point or target point of registation, shape like (N, num_of_correspondence, 3)
        :return:
        """
        pass