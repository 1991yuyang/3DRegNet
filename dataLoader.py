from torch.utils import data
import torch as t
from torch import nn
import open3d as o3d
import numpy as np
from numpy import random as rd
from colors import COLOR_MAP
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

    def __init__(self, voxel_size, R_range, t_range):
        """

        :param voxel_size: voxel size for downsample one point cloud
        :param R_range: value range of item of rotation matrix
        :param t_range: value range of item of translate vector
        """
        self.voxel_size = voxel_size
        self.R_range = R_range
        self.t_range = t_range

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

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
            pcd.paint_uniform_color(COLOR_MAP[40 - i])
        o3d.visualization.draw_geometries(pcd_list, "pcd")

    def random_generate_R_t(self):
        """
        random generate rotation matrix and translate vector
        :return:
        """
        random_lie_param = rd.uniform(self.R_range[0], self.R_range[1], (3, 1))
        R = t.tensor(o3d.geometry.get_rotation_matrix_from_axis_angle(random_lie_param)).type(t.FloatTensor)  # rotation matrix, centroid is the center of rotation
        t_vec = t.tensor(rd.uniform(self.t_range[0], self.t_range[1], 3)).type(t.FloatTensor)
        return R, t_vec

    def rotate_pcd(self, pcd, R):
        """

        :param pcd: point cloud data
        :param R: rotation matrix, shape is (3, 3)
        :return:
        """
        pcd_after_rotate = pcd.rotate(R)
        return pcd_after_rotate

    def translate_pcd(self, pcd, t):
        """

        :param pcd: point cloud data
        :param t: translate vector, shape is (3,)
        :return:
        """
        pcd_after_translate = pcd.translate(t)
        return pcd_after_translate


if __name__ == "__main__":
    from copy import deepcopy
    pcd_pth = r"F:\python_project\test_open3d\cloud_image_00000.ply"
    s = MySet(0.001, [-10, 10], [-10, 10])
    pcd = s.load_one_pc(pcd_pth)
    pcd, fpfh = s.preprocess_point_cloud(pcd)
    pcd_orig = deepcopy(pcd)
    R, t_vec = s.random_generate_R_t()
    pcd_after_rotation = s.rotate_pcd(pcd, R)
    pcd_after_translate = s.translate_pcd(pcd_after_rotation, t_vec)
    s.show_pcd([pcd_orig, pcd_after_translate])