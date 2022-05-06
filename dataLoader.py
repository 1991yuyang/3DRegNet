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

    def __init__(self, data_dir, voxel_size, R_range, t_range, select_point_count, noise_strength, feature_distance_tresh):
        """

        :param data_dir: ply data dir
        :param voxel_size: voxel size for downsample one point cloud
        :param R_range: value range of item of rotation matrix
        :param t_range: value range of item of translate vector
        :param select_point_count: point count of one correspondence
        :param noise_strenght: float, specify the noise strenght while add gaussian noise to source pcd
        :param feature_distance_tresh: feature distance threshold, use for label inlier or outlier
        """
        self.voxel_size = voxel_size
        self.R_range = R_range
        self.t_range = t_range
        self.select_point_count = select_point_count
        self.pcd_pths = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        self.noise_strength = noise_strength
        self.feature_distance_thresh = feature_distance_tresh

    def __getitem__(self, index):
        pcd_pth = self.pcd_pths[index]
        match_indices = np.array([])
        while match_indices.shape[0] == 0:
            pcd = self.load_one_pcd(pcd_pth)
            source = copy.deepcopy(pcd)
            target, R, t_vec = self.random_transform(pcd)  # R * target + t_vec
            source = self.add_gauss_noise_to_pcd(source)  # add noise to source
            target_down, target_down_fpfh = self.preprocess_point_cloud(target)
            source_down, source_down_fpfh = self.preprocess_point_cloud(source)
            fpfh_regis_result = self.execute_global_registration(source_down, target_down, source_down_fpfh, target_down_fpfh)
            match_indices = np.asarray(fpfh_regis_result.correspondence_set)
        target_down_array = np.asarray(target_down.points)
        source_down_array = np.asarray(source_down.points)
        target_down_fpfh_array = np.asarray(target_down_fpfh.data).transpose()
        source_down_fpfh_array = np.asarray(source_down_fpfh.data).transpose()
        target_select, source_select, target_fpfh, source_fpfh = self.select_point_feature(target_down_array, source_down_array, target_down_fpfh_array, source_down_fpfh_array, match_indices)
        feature_dist = self.calc_feature_dist(target_fpfh, source_fpfh)
        class_label = feature_dist <= self.feature_distance_thresh
        # tar = o3d.geometry.PointCloud()
        # sou = o3d.geometry.PointCloud()
        # tar.points = o3d.utility.Vector3dVector(target_select)
        # sou.points = o3d.utility.Vector3dVector(source_select)
        # source_down.transform(fpfh_regis_result.transformation)
        # self.show_pcd([sou, tar])
        d = t.tensor(np.concatenate([source_select, target_select], axis=1)).type(t.FloatTensor)
        class_label = t.tensor(class_label).type(t.FloatTensor)
        return d, class_label, feature_dist

    def __len__(self):
        return len(self.pcd_pths)

    def select_point_feature(self, target_down_array, source_down_array, target_down_fpfh_array, source_down_fpfh_array, match_indices):
        select_indices = match_indices[rd.choice(np.arange(match_indices.shape[0]), self.select_point_count, replace=True), :]
        target_select = target_down_array[select_indices[:, 1], :]
        source_select = source_down_array[select_indices[:, 0], :]
        target_fpfh = target_down_fpfh_array[select_indices[:, 1], :]
        source_fpfh = source_down_fpfh_array[select_indices[:, 0], :]
        return target_select, source_select, target_fpfh, source_fpfh

    def calc_feature_dist(self, target_fpfh, source_fpfh):
        feature_dist = np.sqrt(np.sum((target_fpfh - source_fpfh) ** 2, axis=1))
        return feature_dist

    def add_gauss_noise_to_pcd(self, pcd):
        point = np.asarray(pcd.points)
        signal = point
        SNR = 1 / self.noise_strength * 10  # value为指定噪声强度
        noise = np.random.randn(signal.shape[0], signal.shape[1])
        noise = noise - np.mean(noise)
        signal_power = np.linalg.norm(signal) ** 2 / signal.size
        noise_variance = signal_power / np.power(10, (SNR / 10))
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
        signal_noise = noise + signal
        pcd_noise = o3d.geometry.PointCloud()
        pcd_noise.points = o3d.utility.Vector3dVector(signal_noise)
        return pcd_noise

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

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
       distance_threshold = self.voxel_size * 1.5
       # print(":: RANSAC registration on downsampled point clouds.")
       # print("   Since the downsampling voxel size is %.3f," % self.voxel_size)
       # print("   we use a liberal distance threshold %.3f." % distance_threshold)
       result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
           source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
           o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
               o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
               o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                   distance_threshold)
           ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
       return result


def find_feature_dist_thresh(pcd_dir, voxel_size, R_range, t_range, select_point_count, noise_strength):
    s = MySet(pcd_dir, voxel_size, R_range, t_range, select_point_count, noise_strength, 0.1)
    dist_med_lst = []
    for _, _, feature_dist in s:
        dist_med = np.median(feature_dist)
        dist_med_lst.append(dist_med)
    return np.mean(dist_med_lst)


if __name__ == "__main__":
    pcd_dir = r"F:\python_project\test_open3d\pcd_dir"
    voxel_size = 0.01
    R_range = [-3.14, 3.14]
    t_range = [-1, 1]
    s = MySet(pcd_dir, voxel_size, R_range, t_range, 3000, 0.12, 45)
    # for i in range(100):
    #     s[0]
    for i in range(10):
        print(find_feature_dist_thresh(pcd_dir, voxel_size, R_range, t_range, 3000, 0.12))
