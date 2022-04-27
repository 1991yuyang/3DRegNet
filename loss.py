import torch as t
from torch import nn
from torch.nn import functional as F


class CLSLoss(nn.Module):

    def __init__(self):
        super(CLSLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, use_for_cls_loss, target):
        """

        :param use_for_cls_loss:  cls module features for cls loss caculation, shape like (N, C)
        :param target: ground truth, shape like (N, C), value of item is 0 or 1, 1 represent inlier, 0 represent outlier
        :return:
        """
        target = target.type(t.FloatTensor)
        pos_neg_weights_of_every_sample = t.cat([t.sum(target, dim=1, keepdim=True), t.sum(1 - target, dim=1, keepdim=True)], dim=1) / target.size()[1]
        pred_prob = self.sigmoid(use_for_cls_loss)
        total_loss = 0
        for i in range(use_for_cls_loss.size()[0]):
            pred = pred_prob[i].view((-1, 1))  # shape like (C,)
            tar = target[i].view((-1, 1))  # shape like (C,)
            weight = pos_neg_weights_of_every_sample[i]
            weight_of_curr_sample = t.empty(size=(tar.size()[0],))
            weight_of_curr_sample[(tar == 0).view((-1,))] = weight[0]
            weight_of_curr_sample[(tar == 1).view((-1,))] = weight[1]
            loss = F.binary_cross_entropy(pred, tar, weight=weight_of_curr_sample.view((-1, 1)))
            total_loss += loss
        final_loss = total_loss / use_for_cls_loss.size()[0]
        return final_loss


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, point1, point2):
        """

        :param point1: registration result point, shape like (N, num_of_correspondence, 3)
        :param point2: ground truth point or target point of registation, shape like (N, num_of_correspondence, 3)
        :return:
        """
        pass


if __name__ == "__main__":
    out = t.randn(2, 3)
    tar = t.randint(0, 2, (2, 3))
    model = CLSLoss()
    loss = model(out, tar)
    print(loss)