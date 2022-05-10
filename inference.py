import torch as t
from torch import nn
from model import RefineNet
import open3d as o3d
import os
import numpy as np
from numpy import random as rd
from colors import COLOR_MAP
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


use_best_model = False
threeDRegNet_count = 2
res_block_counts = [16, 8]
num_of_correspondence = 3000
use_lie = True
source_pth = r"source.ply"
target_pth = r"target.ply"
voxel_size = 0.01
M = 3 if use_lie else 9


def preprocess_point_cloud(pcd):
    # print(":: Downsample with a voxel size %.3f." % self.voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
   distance_threshold = voxel_size * 1.5
   # print(":: RANSAC registration on downsampled point clouds.")
   # print("   Since the downsampling voxel size is %.3f," % self.voxel_size)
   # print("   we use a liberal distance threshold %.3f." % distance_threshold)
   result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
       source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
       o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
               distance_threshold)
       ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
   return result


def select_point_feature(target_down_array, source_down_array, target_down_fpfh_array, source_down_fpfh_array,
                         match_indices):
    select_indices = match_indices[rd.choice(np.arange(match_indices.shape[0]), num_of_correspondence, replace=True),
                     :]
    target_select = target_down_array[select_indices[:, 1], :]
    source_select = source_down_array[select_indices[:, 0], :]
    target_fpfh = target_down_fpfh_array[select_indices[:, 1], :]
    source_fpfh = source_down_fpfh_array[select_indices[:, 0], :]
    return target_select, source_select, target_fpfh, source_fpfh


def load_model():
    model = RefineNet(threeDRegNet_count, res_block_counts, num_of_correspondence, M, use_lie)
    model = nn.DataParallel(module=model, device_ids=[0])
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


def load_source_target():
    source = o3d.io.read_point_cloud(source_pth)
    target = o3d.io.read_point_cloud(target_pth)
    target_down, target_down_fpfh = preprocess_point_cloud(target)
    source_down, source_down_fpfh = preprocess_point_cloud(source)
    match_indices = np.array([])
    while match_indices.shape[0] == 0:
        fpfh_regis_result = execute_global_registration(source_down, target_down, source_down_fpfh, target_down_fpfh)
        match_indices = np.asarray(fpfh_regis_result.correspondence_set)
    target_down_array = np.asarray(target_down.points)
    source_down_array = np.asarray(source_down.points)
    target_down_fpfh_array = np.asarray(target_down_fpfh.data).transpose()
    source_down_fpfh_array = np.asarray(source_down_fpfh.data).transpose()
    target_select, source_select, target_fpfh, source_fpfh = select_point_feature(target_down_array,
                                                                                       source_down_array,
                                                                                       target_down_fpfh_array,
                                                                                       source_down_fpfh_array,
                                                                                       match_indices)
    target_select_pcd = o3d.geometry.PointCloud()
    source_select_pcd = o3d.geometry.PointCloud()
    target_select_pcd.points = o3d.utility.Vector3dVector(target_select)
    source_select_pcd.points = o3d.utility.Vector3dVector(source_select)
    show_pcd([source_select_pcd, target_select_pcd])
    d = t.tensor(np.concatenate([source_select, target_select], axis=1)).type(t.FloatTensor).unsqueeze(0).cuda(0)
    return d


def show_pcd(pcd_list):
    """

    :param pcd_list: list of pcd
    :return:
    """
    for i, pcd in enumerate(pcd_list):
        pcd.paint_uniform_color(COLOR_MAP[i])
    o3d.visualization.draw_geometries(pcd_list, "pcd")


def predict(model, d):
    with t.no_grad():
        rotation_mats, trans_mats, cls_outs, reg_outs, use_for_cls_losses, points_preds = model(d)
    points_pred_np = points_preds[-1].squeeze(0).cpu().detach().numpy()
    points_target_np = d[0][:, 3:].cpu().detach().numpy()
    points_pred = o3d.geometry.PointCloud()
    points_target = o3d.geometry.PointCloud()
    points_pred.points = o3d.utility.Vector3dVector(points_pred_np)
    points_target.points = o3d.utility.Vector3dVector(points_target_np)
    show_pcd([points_pred, points_target])


if __name__ == "__main__":
    model = load_model()
    d = load_source_target()
    predict(model, d)

