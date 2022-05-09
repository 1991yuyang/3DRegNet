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
        target = target.type(use_for_cls_loss.dtype).to(use_for_cls_loss.device)
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
            loss = F.binary_cross_entropy(pred, tar, weight=weight_of_curr_sample.view((-1, 1)).to(use_for_cls_loss.device))
            total_loss += loss
        final_loss = total_loss / use_for_cls_loss.size()[0]
        return final_loss


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, points_pred, points_target):
        """

        :param points_pred: registration result point, shape like (N, num_of_correspondence, 3)
        :param points_target: ground truth point or target point of registation, shape like (N, num_of_correspondence, 3)
        :return:
        """
        loss = self.l1(points_pred, points_target)
        return loss


class ClsRegLoss(nn.Module):

    def __init__(self, alpha, beta):
        """

        :param alpha: weight of cls loss
        :param beta: weight of registration loss
        """
        super(ClsRegLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reg = RegLoss()
        self.cls = CLSLoss()

    def forward(self, use_for_cls_loss, cls_target, points_pred, points_target):
        """

        :param use_for_cls_loss: cls module features for cls loss caculation, shape like (N, C)
        :param cls_target: ground truth of classification task, shape like (N, C), value of item is 0 or 1, 1 represent inlier, 0 represent outlier
        :param points_pred: registration result point, shape like (N, num_of_correspondence, 3)
        :param points_target: ground truth point or target point of registation, shape like (N, num_of_correspondence, 3)
        :return:
        """
        cls_loss = self.cls(use_for_cls_loss, cls_target)
        reg_loss = self.reg(points_pred, points_target)
        total_loss = self.alpha * cls_loss + self.beta * reg_loss
        return total_loss


class RefineLoss(nn.Module):

    def __init__(self, alpha, beta):
        """

        :param alpha: weight of cls loss
        :param beta: weight of registration loss
        """
        super(RefineLoss, self).__init__()
        self.cls_reg_loss = ClsRegLoss(alpha, beta)

    def forward(self, use_for_cls_losses, cls_target, points_preds, points_target):
        """

        :param use_for_cls_losses: list of use_for_cls_loss
        :param cls_target: ground truth of classification task, shape like (N, C), value of item is 0 or 1, 1 represent inlier, 0 represent outlier
        :param points_preds: list of points_pred
        :param points_target:
        :return:
        """
        total_loss = 0
        for i in range(len(use_for_cls_losses)):
            use_for_cls_loss = use_for_cls_losses[i]
            points_pred = points_preds[i]
            loss = self.cls_reg_loss(use_for_cls_loss, cls_target, points_pred, points_target)
            total_loss += loss
        avg_loss = total_loss / len(use_for_cls_losses)
        return avg_loss


class CustomLossOptimRt(nn.Module):

    def __init__(self):
        super(CustomLossOptimRt, self).__init__()
        pass

    def forward(self, pred_trans_mats, pred_rot_mats, gt_trans_mat, gt_rot_mat):
        loss_R, _ = RotMatMetric(pred_rot_mats, gt_rot_mat)
        loss_t, _ = TransMetric(pred_trans_mats, pred_rot_mats, gt_trans_mat)
        total = 0.5 * loss_R + 0.5 * loss_t
        return total


def RotMatMetric(pred_rot_mats, gt_rot_mat):
    """

    :param pred_rot_mats: rotation matrix list, shape of item is (N, 3, 3)
    :param gt_rot_mat: ground truth of rotation matrix of every point set, shape is (N, 3, 3)
    :return:
    """
    gt_rot_mat = gt_rot_mat.to(pred_rot_mats[0].device)
    R_final = t.cat([t.eye(3).unsqueeze(0) for i in range(gt_rot_mat.size()[0])], dim=0).type(t.FloatTensor).to(pred_rot_mats[0].device)
    for pred_rot_mat in pred_rot_mats:
        pred_rot_mat = pred_rot_mat  # (N, 3, 3)
        R_final = t.bmm(pred_rot_mat, R_final)
    total_ = 0
    for i in range(R_final.size()[0]):
        pred = R_final[i]
        gt = gt_rot_mat[i]
        m = t.arccos((t.matmul(t.linalg.inv(pred), gt).trace() - 1) / 2)
        total_ += m
    avg = total_ / R_final.size()[0]
    return avg, R_final


def TransMetric(pred_trans_mats, pred_rot_mats, gt_trans_mat):
    """

    :param pred_trans_mats: translation matrix list, shape of item is (N, 3)
    :param pred_rot_mats: rotation matrix list, shape of item is (N, 3, 3)
    :param gt_trans_mat: ground truth of translation matrix, shape is (N, 3)
    :return:
    """
    gt_trans_mat = gt_trans_mat.to(pred_rot_mats[0].device)
    t_final = t.zeros(gt_trans_mat.size()).type(t.FloatTensor).to(pred_rot_mats[0].device)  # (N, 3)
    for i in range(1, len(pred_rot_mats)):
        t_ = pred_trans_mats[i - 1].unsqueeze(1).permute(dims=(0, 2, 1))  # (N, 3, 1)
        for pred_rot_mat in pred_rot_mats[i:]:
            pred_rot_mat = pred_rot_mat  # (N, 3, 3)
            t_ = t.bmm(pred_rot_mat, t_)  # (N, 3, 1)
        t_final += (t_.permute(dims=(0, 2, 1)).squeeze(1))
    t_final += pred_trans_mats[-1]
    m = t.norm(gt_trans_mat - t_final, dim=1).mean()
    return m, t_final


if __name__ == "__main__":
    use_for_cls_loss = [t.randn(2, 256) for i in range(10)]
    cls_target = t.randint(0, 2, (2, 256))
    out = [t.randn(2, 256, 3) for i in range(10)]
    tar = t.randn(2, 256, 3)
    model = RefineLoss(0.5, 0.5)
    loss = model(use_for_cls_loss, cls_target, out, tar)
    print(loss)
