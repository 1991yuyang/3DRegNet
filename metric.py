import torch as t
from torch import nn


def RotMatMetric(pred_rot_mats, gt_rot_mat):
    """

    :param pred_rot_mats: rotation matrix list, shape of item is (N, 3, 3)
    :param gt_rot_mat: ground truth of rotation matrix of every point set, shape is (N, 3, 3)
    :return:
    """
    gt_rot_mat = gt_rot_mat.cpu().detach()
    R_final = t.cat([t.eye(3).unsqueeze(0) for i in range(gt_rot_mat.size()[0])], dim=0).type(t.FloatTensor)
    for pred_rot_mat in pred_rot_mats:
        pred_rot_mat = pred_rot_mat.cpu().detach()  # (N, 3, 3)
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
    t_final = t.zeros(gt_trans_mat.size()).type(t.FloatTensor)  # (N, 3)
    for i in range(1, len(pred_rot_mats)):
        t_ = pred_trans_mats[i - 1].unsqueeze(1).permute(dims=(0, 2, 1)).cpu().detach()  # (N, 3, 1)
        for pred_rot_mat in pred_rot_mats[i:]:
            pred_rot_mat = pred_rot_mat.cpu().detach()  # (N, 3, 3)
            t_ = t.bmm(pred_rot_mat, t_)  # (N, 3, 1)
        t_final += (t_.permute(dims=(0, 2, 1)).squeeze(1))
    t_final += pred_trans_mats[-1].cpu().detach()
    m = t.norm(gt_trans_mat - t_final, dim=1).mean()
    return m, t_final


if __name__ == "__main__":
    pred_rot_mats = [t.randn(10, 3, 3) for i in range(10)]
    pred_trans_mats = [t.randn(10, 3) for i in range(10)]
    gt_trans_mat = t.randn(10, 3)
    gt_rot_mat = t.randn(10, 3, 3)
    rot_metric = RotMatMetric(pred_rot_mats, gt_rot_mat)
    trans_metric = TransMetric(pred_trans_mats, pred_rot_mats, gt_trans_mat)
    print(rot_metric)
    print(trans_metric)