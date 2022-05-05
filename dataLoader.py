from torch.utils import data
import torch as t
from torch import nn
import open3d as o3d
import numpy as np


class MySet(data.Dataset):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def load_one_pc(self, pcd_pth):
        pcd = o3d.io.read_point_cloud(pcd_pth)
        return pcd

    def preprocess_point_cloud(self, pcd):
        # print(":: Downsample with a voxel size %.3f." % self.voxel_size)
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        radius_normal = self.voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def show_pcd(self, pcd):
        pcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([pcd], "pcd")


if __name__ == "__main__":
    pcd_pth = r"F:\python_project\test_open3d\cloud_image_00000.ply"
    s = MySet(0.1)
    pcd = s.load_one_pc(pcd_pth)
    pcd, fpfh = s.preprocess_point_cloud(pcd)
    s.show_pcd(pcd)
