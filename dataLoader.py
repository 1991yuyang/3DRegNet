import copy

from torch.utils import data
import torch as t
from torch import nn
import open3d as o3d
import numpy as np
from numpy import random as rd
from colors import COLOR_MAP
import os
"""
train_data_dir:
    pcd1.ply
    pcd2.ply
    ......
valid_data_dir:
    pcd1.ply
    pcd2.ply
    ......
"""


class MySet(data.Dataset):

    def __init__(self, data_dir, voxel_size, R_range, t_range):
        """

        :param data_dir: ply data dir
        :param voxel_size: voxel size for downsample one point cloud
        :param R_range: value range of item of rotation matrix
        :param t_range: value range of item of translate vector
        """
        self.voxel_size = voxel_size
        self.R_range = R_range
        self.t_range = t_range
        self.pcd_pths = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]

    def __getitem__(self, index):
        pcd_pth = self.pcd_pths[index]
        pcd = self.load_one_pcd(pcd_pth)
        target = copy.deepcopy(pcd)
        source, R, t_vec = self.random_transform(pcd)  # R * target + t_vec
        # p1 = (t.matmul(R, t.tensor(np.asarray(target.points)).type(t.FloatTensor).permute(dims=[1, 0])).permute(dims=[1, 0]) + t_vec.view((-1,))).numpy()
        # p = o3d.geometry.PointCloud()
        # p.points = o3d.utility.Vector3dVector(p1)
        # self.show_pcd([target, source, p])

    def __len__(self):
        return len(self.pcd_pths)

    def load_one_pcd(self, pcd_pth):
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

    def pcd2tensor(self, pcd):
        points = np.asarray(pcd.points)
        pcd_tensor = t.tensor(points).type(t.FloatTensor)
        return pcd_tensor

    def show_pcd(self, pcd_list):
        """

        :param pcd_list: list of pcd
        :return:
        """
        for i, pcd in enumerate(pcd_list):
            pcd.paint_uniform_color(COLOR_MAP[i])
        o3d.visualization.draw_geometries(pcd_list, "pcd")

    def random_generate_R_t(self):
        """
        random generate rotation matrix and translate vector
        :return:
        """
        random_lie_param = rd.uniform(self.R_range[0], self.R_range[1], (3, 1))
        R = t.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle(random_lie_param)).type(t.FloatTensor)  # rotation matrix
        t_vec = t.tensor(rd.uniform(self.t_range[0], self.t_range[1], (3, 1))).type(t.FloatTensor)
        return R, t_vec

    def rotate_pcd(self, pcd, R):
        """

        :param pcd: point cloud data
        :param R: rotation matrix, shape is (3, 3)
        :return:
        """
        pcd_after_rotate = pcd.rotate(R, center=(0, 0, 0))
        return pcd_after_rotate

    def translate_pcd(self, pcd, t):
        """

        :param pcd: point cloud data
        :param t: translate vector, shape is (3,)
        :return:
        """
        pcd_after_translate = pcd.translate(t)
        return pcd_after_translate

    def random_transform(self, pcd):
        R, t_vec = self.random_generate_R_t()
        pcd_after_rotate = self.rotate_pcd(pcd, R)
        pcd_after_translate = self.translate_pcd(pcd_after_rotate, t_vec)
        return pcd_after_translate, R, t_vec


if __name__ == "__main__":
    from copy import deepcopy
    pcd_dir = r"F:\python_project\test_open3d\pcd_dir"
    voxel_size = 0.01
    R_range = [-3.14, 3.14]
    t_range = [-10, 10]
    s = MySet(pcd_dir, voxel_size, R_range, t_range)
    s[0]
