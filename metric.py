import torch as t
from torch import nn


def RotMatMetric(pred_rot_mats, gt_rot_mat):
    """

    :param pred_rot_mats: rotation matrix list, shape of item is (N, 3, 3)
    :param gt_rot_mat: ground truth of rotation matrix of every point set, shape is (N, 3, 3)
    :return:
    """
    total = 0
    gt_rot_mat = gt_rot_mat.cpu().detach()
    for pred_rot_mat in pred_rot_mats:
        pred_rot_mat = pred_rot_mat.cpu().detach()  # (N, 3, 3)
        total_ = 0
        for i in range(pred_rot_mat.size()[0]):
            pred = pred_rot_mat[i]
            gt = gt_rot_mat[i]
            m = t.arccos((t.matmul(t.linalg.inv(pred), gt).trace() - 1) / 2)
            total_ += m
        avg = total_ / pred_rot_mat.size()[0]
        total += avg
    final = total / len(pred_rot_mats)
    return final


def TransMetric(pred_trans_mats, gt_trans_mat):
    """

    :param pred_trans_mats: translation matrix list, shape of item is (N, 3)
    :param gt_trans_mat: ground truth of translation matrix, shape is (N, 3)
    :return:
    """
    total = 0
    gt_trans_mat = gt_trans_mat.cpu().detach()
    for pred_trans_mat in pred_trans_mats:
        pred_trans_mat = pred_trans_mat.cpu().detach()
        m = t.norm(gt_trans_mat - pred_trans_mat, dim=1).mean()
        total += m
    final = total / len(pred_trans_mats)
    return final