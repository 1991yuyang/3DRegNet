from torch.utils import data
import torch as t
from torch import nn
import open3d as o3d
import numpy as np


class MySet(data.Dataset):

    def __init__(self, voxel_size):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def load_one_pc(self, pc_pth):
        pcd = o3d.io.read_point_cloud(pc_pth)
        return pcd

    def preprocess_point_cloud(self, pcd, voxel_size):
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh