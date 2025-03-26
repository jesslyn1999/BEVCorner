import os
import random

import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from bevdepth.utils.visualization import visualize_from_image_arr

from nuscenes.utils.geometry_utils import view_points

from typing import List, Tuple, Union
import math

from shapely.geometry import MultiPoint, box

from bevdepth.utils.visualization import visualize_image_with_bboxes_and_points, visualize_heatmap_per_cam_obj

from bevdepth.layers.heads.head.ops.heatmap_coder import (
	gaussian_radius,
	draw_umich_gaussian,
	draw_gaussian_1D,
	draw_ellip_gaussian,
	draw_umich_gaussian_2D,
)


from bevdepth.layers.heads.head.ops.structures.params_3d import ParamsList

from .nusc_det_util import Calibration



__all__ = ['NuscDetDataset']

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def get_rot(angle):
    """Generate a rotation matrix for the given angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    return torch.Tensor([[c, s], [-s, c]])


def img_transform(img: Image, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img_ori = img.copy()
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat, (resize, resize_dims, crop, flip, rotate), img_ori


"""
def gt_box_img_transform(gt_2d_points, ori_dims, resize_dims, crop, flip, rotate):
    orig_width, orig_height = ori_dims
    new_width, new_height = resize_dims
    left, bottom, right, upper = crop

    # Scale points for resizing
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    resized_points = [(x * scale_x, y * scale_y) for x, y in gt_2d_points]
    
    # Adjust points for cropping
    adjusted_points = [(x - left, y - bottom) for x, y in resized_points]
    
    if flip:
        flipped_points = [((right - left) - x, y) for x, y in adjusted_points]
    else:
        flipped_points = adjusted_points.copy()
    
    # Rotate
    
    cx, cy = (right - left) / 2, (upper - bottom) / 2
    translated_points = [(x - cx, y - cy) for x, y in flipped_points]
    
    angle_rad = rotate / 180 * np.pi  # Convert degrees to radians
    rotation_matrix = get_rot(angle_rad)
    rotated_points = np.array(translated_points) @ rotation_matrix.T.detach().cpu().numpy()  # Rotate points
    
    translated_rotated_points = [(x + cx, y + cy) for x, y in rotated_points]

    # # Translate points back after cropping
    # transformed_points = [(x + left, y + bottom) for x, y in rotated_points]
    
    return translated_rotated_points

"""

def gt_box_img_transform(gt_2d_points, ori_dims, resize_dims, crop, flip, rotate):
    orig_width, orig_height = ori_dims
    new_width, new_height = resize_dims
    left, bottom, right, upper = crop

    # Scale points for resizing
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    # Convert points to numpy array for easier manipulation
    points = np.array(gt_2d_points)
    
    # Scale
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    
    # Crop
    points[:, 0] -= left
    points[:, 1] -= bottom
    
    # Flip
    if flip:
        points[:, 0] = (right - left) - points[:, 0]
    
    # Rotate around center
    if rotate != 0:
        cx = (right - left) / 2
        cy = (upper - bottom) / 2
        
        # Translate to origin
        points[:, 0] -= cx
        points[:, 1] -= cy
        
        # Rotate
        angle_rad = rotate / 180 * np.pi
        rot_mat = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])
        points = points @ rot_mat.T
        
        # Translate back
        points[:, 0] += cx
        points[:, 1] += cy
    
    return points.tolist()


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def depth_transform_v1(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """
    cam_depth = cam_depth.float()
    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = torch.tensor(rotate / 180 * np.pi)

    # Given rotation matrix in PyTorch
    rot_matrix = torch.tensor([
        [torch.cos(h), torch.sin(h)],
        [-torch.sin(h), torch.cos(h)]
    ])
    cam_depth[:, :2] = torch.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].numpy().astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                    & (depth_coords[:, 0] < resize_dims[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def depth_transform_3d(depth_map_2d, depth_channel):
    # add depth dimension
    # input H * W
    # return D * H * W
    H, W = depth_map_2d.shape
    depths = torch.where(
            (depth_map_2d < depth_channel) & (depth_map_2d >= 0.0),
            depth_map_2d, torch.zeros_like(depth_map_2d))
    depth_map_3d = torch.zeros_like(depth_map_2d)
    depth_map_3d = depth_map_3d.unsqueeze(0).expand(depth_channel, H, W)
    depth_map_3d = depth_map_3d.clone()

    h_indices, w_indices = torch.nonzero(depths, as_tuple=True)
    d_indices = depths[h_indices, w_indices].long()
    depth_map_3d[d_indices, h_indices, w_indices] = 1
    return depth_map_3d


def map_pointcloud_to_image(
    lidar_points,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


class NuscDetDataset(Dataset):

    def __init__(self,
                ida_aug_conf,
                bda_aug_conf,
                classes,
                data_root,
                info_paths,
                is_train,
                use_cbgs=False,
                num_sweeps=1,
                img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                            img_std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                return_depth=False,
                sweep_idxes=list(),
                key_idxes=list(),
                use_fusion=False,
                filter_params=None,
                down_ratio=None,
                orientation_method=None,
                multibin_size=None,
                edge_heatmap_ratio=None,
                width_train=None,
                height_train=None,
                depth_channel=None,
                is_bev=True,
                is_monocular=True,
                ):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            bda_aug_conf (dict): Config for bda augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
            use_fusion (bool): Whether to use lidar data.
                default: False.
        """
        super().__init__()
        if isinstance(info_paths, list):
            self.infos = list()
            for info_path in info_paths:
                self.infos.extend(mmcv.load(info_path))
        else:
            self.infos = mmcv.load(info_paths)
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.return_depth = return_depth
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) \
            == len(sweep_idxes), 'All `sweep_idxes` must greater \
                than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(key_idxes),\
            'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes
        self.use_fusion = use_fusion
        self.filter_params = filter_params
        self.down_ratio = down_ratio
        
        PI = np.pi
        self.orientation_method = orientation_method
        self.multibin_size = multibin_size
        self.alpha_centers = np.array([0, PI / 2, PI, - PI / 2]) # centers for multi-bin orientation
        self.edge_heatmap_ratio = edge_heatmap_ratio

        # final dim: 256, 704
        self.output_width = width_train // down_ratio  # 704 / 16 = 44  # 704 / 4 = 176
        self.output_height = height_train // down_ratio  # 256 / 16 = 16  # 256 / 4 = 64
        
        self.max_objs = 200
        self.enable_edge_fusion = True
        self.max_edge_length = (self.output_width + self.output_height) * 2  # (176 + 64) * 2 = 480

        self.depth_channel = depth_channel

        self.is_bev=True
        self.is_monocular=True


    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
        lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        pts_img, depth = map_pointcloud_to_image(
            lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                              axis=1).astype(np.float32)
    
    def get_obj_lvl_depth(self, centers_3d, box2d):
        # get the coordinates of all pixels in bbox
        min_x, min_y, max_x, max_y = box2d
        min_x, min_y, max_x, max_y = np.int(min_x), np.int(min_y), np.int(max_x), np.int(max_y) 
        depth = centers_3d[-1]

        x_size = max_x-min_x
        y_size = max_y-min_y
        x_coords = torch.arange(min_x, max_x, 1, dtype=torch.int).view(1, x_size).expand(y_size, x_size)
        y_coords = torch.arange(min_y, max_y, 1, dtype=torch.int).view(y_size, 1).expand(y_size, x_size)
        depth_coords = torch.tensor(depth).unsqueeze(-1).expand(y_size, x_size)

        # n_points x 3
        grid = torch.stack((x_coords, y_coords, depth_coords), -1)
        return grid.reshape(-1, 3)


    def get_image(self, cam_infos, cams, lidar_infos=None):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        ori_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        
        sweep_ida_augs = list()
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            for lidar_info in lidar_infos:
                lidar_path = lidar_info['LIDAR_TOP']['filename']
                lidar_points = np.fromfile(os.path.join(
                    self.data_root, lidar_path),
                                           dtype=np.float32,
                                           count=-1).reshape(-1, 5)[..., :4]
                sweep_lidar_points.append(lidar_points)
        for cam in cams:
            imgs = list()
            ori_imgs_sub = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            cam2ego_per_cam_rotations = list()

            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
                
            ida_augs = list()

            for sweep_idx, cam_info in enumerate(cam_infos):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                
                cam2ego_per_cam_rotation = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['rotation'])

                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)
                img, ida_mat, ida_aug, img_ori = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                ida_augs.append(ida_aug)
                ori_imgs_sub .append(img_ori)
                intrin_mats.append(intrin_mat)
                cam2ego_per_cam_rotations.append(cam2ego_per_cam_rotation)
                timestamps.append(cam_info[cam]['timestamp'])
                


            sweep_imgs.append(torch.stack(imgs))
            ori_imgs.append(ori_imgs_sub)
            sweep_ida_augs.append(ida_augs)

            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))

        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))

        return ret_list, sweep_ida_augs, ori_imgs
    
    
    

    def get_gt(self, info, cams,
               sweep_imgs=[], sweep_ida_augs=[], ori_imgs=[], is_bev=True,is_monocular=True):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        imsize: Tuple[int, int] = ori_imgs[0][0].size
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        
        ego2global_per_cam_rotations = [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams]
        ego2global_per_cam_trans = [info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams]

        ego2egocam_per_cam_rotations = [info['cam_infos'][cam]['calibrated_sensor']['rotation'] for cam in cams]
        ego2egocam_per_cam_trans = [info['cam_infos'][cam]['calibrated_sensor']['translation'] for cam in cams]
        
        intrins_per_cam = [np.array(info['cam_infos'][cam]['calibrated_sensor']['camera_intrinsic']) for cam in cams]

        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()

        # the boundaries of the image after padding
        x_min, y_min = 0, 0
        x_max, y_max = self.output_width, self.output_height

        if self.enable_edge_fusion:
            # generate edge_indices for the edge fusion module
            input_edge_indices = np.zeros([len(cams), self.max_edge_length, 2], dtype=np.int64)
            
            input_edge_counts = np.zeros([len(cams)], dtype=np.int64)

            edge_indices = self.get_edge_utils((self.output_width, self.output_height)).numpy()
            input_edge_count = edge_indices.shape[1]
            input_edge_indices[:, :edge_indices.shape[0]] = edge_indices.copy()
            input_edge_count = input_edge_count - 1 # explain ? 
            input_edge_counts.fill(input_edge_count)
    

        calibs = [self.get_calibration(intrins) for intrins in intrins_per_cam]

        if False and not self.is_train: # TODO: uncomment
            # for inference we parametrize with original size
            target = ParamsList(image_size=ori_imgs[0][0].size, 
                                is_train=self.is_train, 
                                resize_dims=self.ida_aug_conf['final_dim'][::-1], 
                                resize_dims_mini=[self.output_width, self.output_height])

            target.add_field("pad_size", np.tile(np.array([0, 0]), (len(cams), 1)))
            target.add_field("calib", calibs)

            if self.enable_edge_fusion:
                target.add_field('edge_len', input_edge_counts)
                target.add_field('edge_indices', input_edge_indices)

            return torch.Tensor(gt_boxes), torch.tensor(gt_labels), target, None

        # heatmap
        heat_map = np.zeros([len(cams), len(self.classes), self.output_height, self.output_width], dtype=np.float32)

        # classification
        cls_ids = np.zeros([len(cams), self.max_objs], dtype=np.int32) 
        target_centers = np.zeros([len(cams), self.max_objs, 2], dtype=np.int32)
        # 2d bounding boxes
        gt_bboxes = np.zeros([len(cams), self.max_objs, 4], dtype=np.float32)
        bboxes = np.zeros([len(cams), self.max_objs, 4], dtype=np.float32)
        # keypoints: 2d coordinates and visible(0/1)
        keypoints = np.zeros([len(cams), self.max_objs, 10, 3], dtype=np.float32)
        keypoints_depth_mask = np.zeros([len(cams), self.max_objs, 3], dtype=np.float32) # whether the depths computed from three groups of keypoints are valid
        keypoints_scale = np.zeros([len(cams), self.max_objs], dtype=np.float32)
        # 3d dimension
        dimensions = np.zeros([len(cams), self.max_objs, 3], dtype=np.float32)
        # 3d location
        locations = np.zeros([len(cams), self.max_objs, 3], dtype=np.float32)
        # rotation y
        rotys = np.zeros([len(cams), self.max_objs], dtype=np.float32)
        # alpha (local orientation)
        alphas = np.zeros([len(cams), self.max_objs], dtype=np.float32)
        # offsets from center to expected_center
        offset_3D = np.zeros([len(cams), self.max_objs, 2], dtype=np.float32)

        # occlusion and truncation
        occlusions = np.zeros([len(cams), self.max_objs])
        truncations = np.zeros([len(cams), self.max_objs])

        depth_obj_lvl_maps = torch.zeros((len(cams), self.depth_channel,
                                          self.ida_aug_conf['final_dim'][0], self.ida_aug_conf['final_dim'][1]))  # Consist of 3d depth map D * H * W

		
        if self.orientation_method == 'head-axis': orientations = np.zeros([len(cams), self.max_objs, 3], dtype=np.float32)
        else: orientations = np.zeros([len(cams), self.max_objs, self.multibin_size * 2], dtype=np.float32) # multi-bin loss: 2 cls + 2 offset

        reg_mask = np.zeros([len(cams), self.max_objs], dtype=np.uint8) # regression mask
        trunc_mask = np.zeros([len(cams), self.max_objs], dtype=np.uint8) # outside object mask
        reg_weight = np.zeros([len(cams), self.max_objs], dtype=np.float32) # regression weight

        for ann_idx, ann_info in enumerate(info['ann_infos']):
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )

            # global to ego. hm why?
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
            
            
            ## additional geometric groundtruths
            # 8 keypoints (corners) + 2 keypoints (top-center, bottom-center)
            
            cls_id = self.classes.index(map_name_from_general_to_detection[ann_info['category_name']])
            if cls_id < 0: continue

            for idx, cam in enumerate(cams):

                box = Box(
                    ann_info['translation'],
                    ann_info['size'],
                    Quaternion(ann_info['rotation']),
                    velocity=ann_info['velocity'],
                )

                ori_img = ori_imgs[idx][0]
                sweep_img = visualize_from_image_arr(sweep_imgs[0][idx])
                # Move them to the ego-pose frame.
                box.translate(-np.array(ego2global_per_cam_trans[idx]))
                box.rotate(Quaternion(ego2global_per_cam_rotations[idx]).inverse)

                # Move them to the calibrated sensor frame.
                box.translate(-np.array(ego2egocam_per_cam_trans[idx]))
                box.rotate(Quaternion(ego2egocam_per_cam_rotations[idx]).inverse)

                # Filter out the corners that are not in front of the calibrated sensor.
                corners_3d = box.corners()
                centers_3d = box.center
                centers_3d_bottom = centers_3d.copy()
                centers_3d_bottom[1] = centers_3d_bottom[1] - box.wlh[-1] / 2

                in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                corners_3d_front = corners_3d[:, in_front]

                # Project 3d box to 2d.
                corner_coords = np.array(view_points(corners_3d, intrins_per_cam[idx], True).T[:, :2])
                corner_coords_front = np.array(view_points(corners_3d_front, intrins_per_cam[idx], True).T[:, :2])
                # center_coords = np.array(view_points(np.array([centers_3d_bottom]).T, intrins_per_cam[idx], True).T[:, :2])[0]
                
                # unlike using kitti dataset in monoflex, we didn't regards center bottom as the real center
                center_coords = np.array(view_points(np.array([centers_3d.copy()]).T, intrins_per_cam[idx], True).T[:, :2])[0]

                # Keep only corners that fall within the image.
                final_coords = post_process_coords(corner_coords_front, imsize=imsize)


                # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
                if final_coords is None or centers_3d[2] <= 0:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords
                
                final_coords = np.array(final_coords)
                float_occlusion = float(1) # 0 for normal, 0.33 for partially, 0.66 for largely, 1 for unknown (mostly very far and small objs)
                float_truncation = 1 # 0 ~ 1 and stands for truncation level

                locs = centers_3d.copy()
                locs[1] = locs[1] - box.wlh[-1] / 2  # bottom centers ==> 3D centers
                if locs[-1] <= 0: continue # objects which are behind the image

                center_coords_bottom_center = np.array(view_points(np.array([locs.copy()]).T, intrins_per_cam[idx], True).T[:, :2])[0]


                # generate 8 corners of 3d bbox
                corners_3d = corners_3d.T[[3,7,6,2,0,4,5,1]]  # dump: 3, 7, 6, 2, 0, 4, 5, 1
                corners_2d = corner_coords[[3,7,6,2,0,4,5,1]]
                projected_box2d = np.array([corners_2d[:, 0].min(), corners_2d[:, 1].min(), 
                                            corners_2d[:, 0].max(), corners_2d[:, 1].max()])
                # TODO: debug if needed. visualize_image_with_bboxes_and_points(ori_img, t_bboxes=[box2d], t_points=corner_coords)
                box2d = final_coords.copy()

                resize, resize_dims, crop, flip, rotate = sweep_ida_augs[idx][0]

                if self.return_depth:
                    point_depth = self.get_obj_lvl_depth(centers_3d, box2d)
                    point_depth_augmented = depth_transform_v1(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate)
                    depth_map_3d = depth_transform_3d(point_depth_augmented , self.depth_channel)
                    
                    tmp_mask = depth_obj_lvl_maps[idx] < depth_map_3d
                    # C, H, W
                    depth_obj_lvl_maps[idx] = torch.where(tmp_mask, depth_map_3d, depth_obj_lvl_maps[idx])
                    
                    # TODO: debug if needed. call visualize_3d_depth(depth_obj_lvl_maps)

                # filter some unreasonable annotations
                if float_truncation >= self.filter_params[0] and (box2d[2:] - box2d[:2]).min() <= self.filter_params[1]: continue

                # project 3d location to the image plane
                # proj_center = center_coords.copy()
                proj_center = center_coords_bottom_center.copy()

                # generate approximate projected center when it is outside the image
                proj_inside_img = (0 <= proj_center[0] <= imsize[0] - 1) & (0 <= proj_center[1] <= imsize[1] - 1)

                approx_center = False
                if not proj_inside_img:
                    approx_center = True
                    center_2d = (box2d[:2] + box2d[2:]) / 2
                    target_proj_center, edge_index = approx_proj_center(proj_center, center_2d.reshape(1, 2), (imsize[0], imsize[1]))
                else:
                    target_proj_center = proj_center.copy()

                # 10 keypoints
                bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
                keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
                keypoints_2D = np.array(view_points(keypoints_3D.T, intrins_per_cam[idx], True).T[:, :2])

                # Check visibility
                # keypoints mask: keypoint must be inside the image and in front of the camera
                keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= imsize[0] - 1)
                keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= imsize[1] - 1)
                keypoints_z_visible = (keypoints_3D[:, -1] > 0)

                # Create depth validity mask (for different keypoint groups)
                # center (8,9), diagonal-02 (0,2,4,6), diagonal-13 (1,3,5,7)
                # xyz visible
                keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
                # center, diag-02, diag-13
                keypoints_depth_valid = np.stack([
                    keypoints_visible[[8, 9]].all(), 
                    keypoints_visible[[0, 2, 4, 6]].all(), 
                    keypoints_visible[[1, 3, 5, 7]].all()
                ])

                if True: 
                    keypoints_visible = np.append(np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2), np.tile(keypoints_visible[8] | keypoints_visible[9], 2))
                    keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))

                    keypoints_visible = keypoints_visible.astype(np.float32)
                    keypoints_depth_valid = keypoints_depth_valid.astype(np.float32)
                
                keypoints_2D_raw = keypoints_2D.copy()
                center_coords_raw = center_coords.copy() # before ida aug
                target_proj_center_raw = target_proj_center.copy()
                box2d_raw  = box2d.copy()
                
                # TODO: debug, breakpoint right on this below line with visualization tools if needed
                # can call visualize_image_with_bboxes_and_points(ori_img, t_bboxes=[box2d], t_points=np.append(keypoints_2D, [center_coords, target_proj_center], axis=0))
                # visualize_image_with_bboxes_and_points(ori_img, t_bboxes=[box2d], t_points=[np.append(keypoints_2D, [center_coords, target_proj_center], axis=0)], id="cam{}-ann{}-ori".format(idx, ann_idx))


                # keypoints_2D = keypoints_2D_raw.copy()
                # center_coords = center_coords_raw.copy()
                # target_proj_center = target_proj_center_raw.copy()
                # box2d = box2d_raw.copy()

                
                keypoints_2D_center = gt_box_img_transform(np.append(keypoints_2D, [center_coords, proj_center, target_proj_center], axis=0), 
                                        ori_dims = ori_img.size, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)

                box2d = np.array(gt_box_img_transform(
                    np.append([box2d[:2], box2d[2:]], 
                              [[box2d[0], box2d[3]], [box2d[2], box2d[1]]],   # (xmin, ymax), (xmax, ymin)
                              axis=0),
                    ori_dims = ori_img.size, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate))
                
                # get the largest enclosure
                box2d_xmin, box2d_xmax = box2d[:, 0].min(), box2d[:, 0].max()
                box2d_ymin, box2d_ymax = box2d[:, 1].min(), box2d[:, 1].max()

                box2d = np.array([box2d_xmin, box2d_ymin, box2d_xmax, box2d_ymax])

                keypoints_2D = np.array(keypoints_2D_center[:-3])
                center_coords = np.array(keypoints_2D_center[-3])
                proj_center = np.array(keypoints_2D_center[-2])
                target_proj_center = np.array(keypoints_2D_center[-1])
                
                
                # TODO: debug, breakpoint right on this below line with visualization tools if needed
                # can call visualize_image_with_bboxes_and_points(sweep_img, t_bboxes=[box2d], t_points=np.array(keypoints_2D_center))
                # visualize_image_with_bboxes_and_points(sweep_img, t_bboxes=[box2d], t_points=[np.array(keypoints_2D_center)], id="cam{}-ann{}-sweep".format(idx, ann_idx))
                
                down_ratio_horizontal = (crop[2] - crop[0]) / self.output_width  # final_dim[1] / self.output_width
                down_ratio_vertical = (crop[3] - crop[1]) / self.output_height   # final_dim[0] / self.output_height

                # downsample bboxes, points to the scale of the extracted feature map (stride = 4)
                keypoints_2D[:, 0] = keypoints_2D[:, 0] / down_ratio_horizontal
                keypoints_2D[:, 1] = keypoints_2D[:, 1] / down_ratio_vertical
                
                target_proj_center[0] = target_proj_center[0] / down_ratio_horizontal
                target_proj_center[1] = target_proj_center[1] / down_ratio_vertical

                # proj_center[0] = center_coords[0] / down_ratio_horizontal
                # proj_center[1] = center_coords[1] / down_ratio_vertical
                proj_center[0] = proj_center[0] / down_ratio_horizontal
                proj_center[1] = proj_center[1] / down_ratio_vertical

                box2d[:3:2] = box2d[:3:2] / down_ratio_horizontal
                box2d[1::2] = box2d[1::2] / down_ratio_vertical
                # 2d bbox center and size
                bbox_center = (box2d[:2] + box2d[2:]) / 2
                bbox_dim = box2d[2:] - box2d[:2]

                target_center = target_proj_center.round().astype(np.int)

                # clip to the boundary
                target_center[0] = np.clip(target_center[0], x_min, x_max)
                target_center[1] = np.clip(target_center[1], y_min, y_max)
                
                box2d[:3:2] = np.clip(box2d[:3:2], x_min, x_max)
                box2d[1::2] = np.clip(box2d[1::2], y_min, y_max)


                # TODO: debug, breakpoint right on this below line with visualization tools if needed
                # can call visualize_image_with_bboxes_and_points(sweep_img, t_bboxes=[np.array([box2d_xmin, box2d_ymin, box2d_xmax, box2d_ymax])], t_points=np.array(keypoints_2D_center))
                
                # visualize_image_with_bboxes_and_points(sweep_img.resize((self.output_width, self.output_height)), 
                #                                        t_bboxes=[np.array(box2d)], 
                #                                        t_points=np.append(keypoints_2D, [proj_center, target_proj_center], axis=0),
                #                                         id="cam{}-ann{}-sweep_resize".format(idx, ann_idx))


                pred_2D = True # In fact, there are some wrong annotations where the target center is outside the box2d
                # if not (target_center[0] >= box2d[0] and target_center[1] >= box2d[1] and target_center[0] <= box2d[2] and target_center[1] <= box2d[3]):
                #     pred_2D = False

                # improvise give a slight gap
                if not (target_center[0] >= box2d[0]-1 and target_center[1] >= box2d[1]-1 and target_center[0] <= box2d[2]+1 and target_center[1] <= box2d[3]+1):
                    pred_2D = False

                if (bbox_dim > 0).all() and (0 <= target_center[0] <= self.output_width - 1) and (0 <= target_center[1] <= self.output_height - 1):
                    rot_y = box.orientation.yaw_pitch_roll[0]
                    
                    # ensure yaw is within -phi, phi
                    rot_y = (rot_y + np.pi) % (2*np.pi) - np.pi
                    alpha = convertRot2Alpha(rot_y, centers_3d[2], centers_3d[0])

                    # generating heatmap
                    if approx_center:
                        # for outside objects, generate 1-dimensional heatmap
                        bbox_width = min(target_center[0] - box2d[0], box2d[2] - target_center[0])
                        bbox_height = min(target_center[1] - box2d[1], box2d[3] - target_center[1])
                        radius_x, radius_y = bbox_width * self.edge_heatmap_ratio, bbox_height * self.edge_heatmap_ratio
                        radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
                        # assert min(radius_x, radius_y) == 0
                        heat_map[idx][cls_id] = draw_umich_gaussian_2D(heat_map[idx][cls_id], target_center, radius_x, radius_y)	
                        
                        
                        # heat_map_vis = np.zeros([self.output_height, self.output_width], dtype=np.float32)
                        # heat_map_vis = draw_umich_gaussian_2D(heat_map_vis, target_center, radius_x, radius_y)					
                    else:
                        # for inside objects, generate circular heatmap
                        radius = gaussian_radius(bbox_dim[1], bbox_dim[0])
                        radius = max(0, int(radius))
                        heat_map[idx][cls_id] = draw_umich_gaussian(heat_map[idx][cls_id], target_center, radius)
                        
                        # heat_map_vis = np.zeros([self.output_height, self.output_width], dtype=np.float32)
                        # heat_map_vis = draw_umich_gaussian(heat_map_vis, target_center, radius)
        
                    # TODO: debug, breakpoint right on this below line with visualization tools if needed
                    # can call visualize_heatmap_per_cam_obj(sweep_img, t_bboxes=[np.array([box2d_xmin, box2d_ymin, box2d_xmax, box2d_ymax])], t_points=np.array(keypoints_2D_center))
                    # visualize_heatmap_per_cam_obj(heat_map_vis, id="cam{}-ann{}-heatmap".format(idx, ann_idx))
                    # visualize_heatmap_per_cam_obj(heat_map[idx][cls_id], id="cam{}-cls{}-heatmap".format(idx, cls_id))

                    cls_ids[idx][ann_idx] = cls_id
                    target_centers[idx][ann_idx] = target_center
                    # offset due to quantization for inside objects or offset from the interesection to the projected 3D center for outside objects
                    offset_3D[idx][ann_idx] = proj_center - target_center
                    
                    # 2D bboxes
                    gt_bboxes[idx][ann_idx] = final_coords # for visualization
                    if pred_2D: bboxes[idx][ann_idx] = box2d  # xmin, ymin, xmax, ymax

                    # local coordinates for keypoints
                    keypoints[idx][ann_idx] = np.concatenate((keypoints_2D - target_center.reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)
                    
                    if False:
                        # Normalize keypoint coordinates relative to center
                        keypoints_2D_centered = keypoints_2D - target_center.reshape(1, -1)

                        # Scale normalization (optional, but helps with training stability)
                        scale = np.sqrt((keypoints_2D_centered ** 2).sum(axis=1)).mean()
                        keypoints_data_scale = scale
                        if scale > 0:
                            keypoints_2D_centered = keypoints_2D_centered / scale

                        keypoints[idx][ann_idx] = np.concatenate((keypoints_2D_centered, keypoints_visible[:, np.newaxis]), axis=1)
                        keypoints_scale[idx][ann_idx] = keypoints_data_scale
                    
                    keypoints_depth_mask[idx][ann_idx] = keypoints_depth_valid
                    
                    dimensions[idx][ann_idx] = np.array(box.wlh[[1, 2, 0]])
                    locations[idx][ann_idx] = locs
                    rotys[idx][ann_idx] = rot_y
                    alphas[idx][ann_idx] = alpha

                    orientations[idx][ann_idx] = self.encode_alpha_multibin(alpha, num_bin=self.multibin_size)

                    reg_mask[idx][ann_idx] = 1
                    reg_weight[idx][ann_idx] = 1 # all objects are of the same weights (for now)
                    trunc_mask[idx][ann_idx] = int(approx_center) # whether the center is truncated and therefore approximate
                    occlusions[idx][ann_idx] = float_occlusion
                    truncations[idx][ann_idx] = float_truncation


        target = ParamsList(image_size=ori_imgs[0][0].size, is_train=self.is_train, resize_dims=self.ida_aug_conf['final_dim'][::-1]
                            , resize_dims_mini=[self.output_width, self.output_height])
        
        
        # TODO: debug steps
        # idx = 2  <-- cam length
        # depth_obj_lvl_maps[idx][depth_obj_lvl_maps[idx] > 0].shape
        # run visualize_3d_depth(depth_obj_lvl_maps[idx])
        # see the ori imgs and compared, run ori_imgs[idx][0].save('depth_{}_cam.png.png'.format(idx))

        # monoflex input is resized dimension
        
        
        target.add_field("cls_ids", cls_ids)  # turn out it is resized coordinates
        target.add_field("target_centers", target_centers)
        target.add_field("keypoints", keypoints)
        target.add_field("keypoints_depth_mask", keypoints_depth_mask)

        if False:
            target.add_field("keypoints_scale", keypoints_scale)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("reg_mask", reg_mask)
        target.add_field("reg_weight", reg_weight)
        target.add_field("offset_3D", offset_3D)
        target.add_field("2d_bboxes", bboxes)
        target.add_field("rotys", rotys)
        target.add_field("trunc_mask", trunc_mask)
        target.add_field("alphas", alphas)
        target.add_field("orientations", orientations)
        target.add_field("hm", heat_map)
        target.add_field("gt_bboxes", gt_bboxes) # for validation visualization
        target.add_field("occlusions", occlusions)
        target.add_field("truncations", truncations)
        
        # for id_cam_vis in range(0, heat_map.shape[0]):
        #     for id_class_vis in range(0, heat_map.shape[1]):
        #         # id_cam_vis = random.randint(0, heat_map.shape[0]-1)
        #         # id_class_vis = random.randint(0, heat_map.shape[1]-1)
        #         visualize_heatmap_per_cam_obj(heat_map[id_cam_vis][id_class_vis], id="cam{}-cls{}-gt-start-heatmap".format(id_cam_vis, id_class_vis))

        target.add_field("pad_size", np.tile(np.array([0, 0]), (len(cams), 1)))
        target.add_field("calib", calibs)

        if self.enable_edge_fusion:
            target.add_field('edge_len', input_edge_counts)
            target.add_field('edge_indices', input_edge_indices)

        return torch.Tensor(gt_boxes), torch.tensor(gt_labels), target, depth_obj_lvl_maps


    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams


    def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
        # encode alpha (-PI ~ PI) to 2 classes and 1 regression
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin # pi
        margin_size = bin_size * margin # pi / 6

        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size

        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset

        return encode_alpha


    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        lidar_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            lidar_infos.append(info['lidar_infos'])
            lidar_sweep_timestamps = [
                lidar_sweep['LIDAR_TOP']['timestamp']
                for lidar_sweep in info['lidar_sweeps']
            ]
            for sweep_idx in self.sweeps_idx:
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                    lidar_infos.append(info['lidar_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx),
                                   -1, -1):
                        if sum([cam in info['cam_sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            cam_timestamp = np.mean([
                                val['timestamp']
                                for val in info['cam_sweeps'][i].values()
                            ])
                            # Find the closest lidar frame to the cam frame.
                            lidar_idx = np.abs(lidar_sweep_timestamps -
                                               cam_timestamp).argmin()
                            lidar_infos.append(info['lidar_sweeps'][lidar_idx])
                            break
        if self.return_depth or self.use_fusion:
            image_data_list, sweep_ida_augs, ori_imgs = self.get_image(cam_infos, cams, lidar_infos)

        else:
            image_data_list, sweep_ida_augs, ori_imgs = self.get_image(cam_infos, cams)
        ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_data_list[:7]
        img_metas['token'] = self.infos[idx]['sample_token']
        
        obj_lvl_depth_maps = None
        if self.is_train:
            gt_boxes, gt_labels, targets, obj_lvl_depth_maps = self.get_gt(self.infos[idx], cams, sweep_imgs=sweep_imgs, sweep_ida_augs=sweep_ida_augs,
                                              ori_imgs=ori_imgs, is_bev=self.is_bev, is_monocular=self.is_monocular)
        # Temporary solution for test.
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )
            
            _, _, targets, _ = self.get_gt(self.infos[idx], cams, sweep_imgs=sweep_imgs, sweep_ida_augs=sweep_ida_augs,
                                              ori_imgs=ori_imgs)

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        ret_list = [None] * 13

        ret_list[0] = sweep_imgs

        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
            None,
            None, 
            targets,
        ]
        if self.return_depth:
            ret_list[-3] = image_data_list[7]
            ret_list[-2] = obj_lvl_depth_maps
        return ret_list


    def get_edge_utils(self, output_size):
        x_min, y_min = 0, 0
        x_max, y_max = output_size[0] - 1, output_size[1] - 1

        step = 1
        # boundary idxs
        edge_indices = []

        # left
        y = torch.arange(y_min, y_max, step)
        x = torch.ones(len(y)) * x_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = torch.arange(x_min, x_max, step)
        y = torch.ones(len(x)) * y_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = torch.arange(y_max, y_min, -step)
        x = torch.ones(len(y)) * x_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # top  
        x = torch.arange(x_max, x_min - 1, -step)
        y = torch.ones(len(x)) * y_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = torch.cat([index.long() for index in edge_indices], dim=0)

        return edge_indices

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)

    def get_calibration(self, intrin_mat):
        return Calibration(intrin_mat)



def collate_fn(data, is_bev=True, is_monocular=True, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    obj_lvl_depth_labels_batch = list()
    target_monocular_batch = list()

    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
            gt_depth,
            gt_obj_lvl_depth,
            target_monocular
        ) = iter_data

        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)

        depth_labels_batch.append(gt_depth)
        obj_lvl_depth_labels_batch.append(gt_obj_lvl_depth)
        target_monocular_batch.append(target_monocular)
        
    
    # for id_b_vis, target_monocular_vis in enumerate(target_monocular_batch):
    #     heat_map = target_monocular_vis.get_field("hm")
    #     for id_cam_vis in range(0, heat_map.shape[0]):
    #         for id_class_vis in range(0, heat_map.shape[1]):
    #             # id_cam_vis = random.randint(0, heat_map.shape[0]-1)
    #             # id_class_vis = random.randint(0, heat_map.shape[1]-1)
    #             visualize_heatmap_per_cam_obj(heat_map[id_cam_vis][id_class_vis], id="batch{}-cam{}-cls{}-gt-start-heatmap".format(id_b_vis, id_cam_vis, id_class_vis))
        
    
    mats_dict = None
    if is_bev: 
        mats_dict = dict()
        mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
        mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
        mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
        mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
        mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    
    ret_list = [
        torch.stack(imgs_batch),
        None,
        None,
        None,
        None,
        None,
    ]

    ret_list = ret_list + [None] * 3

    if is_bev:
        ret_list[1] = mats_dict
        ret_list[2] = torch.stack(timestamps_batch)
        ret_list[3] = img_metas_batch
        ret_list[4] = gt_boxes_batch
        ret_list[5] = gt_labels_batch

        if is_return_depth:
            ret_list[6] = torch.stack(depth_labels_batch)
    
    if is_monocular:
        if is_bev and is_return_depth:
            ret_list[7] = torch.stack(obj_lvl_depth_labels_batch)
        ret_list[8] = target_monocular_batch

    return ret_list


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0] - 1, imsize[1] - 1)

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
    

def approx_proj_center(proj_center, surface_centers, img_size):
    # surface_inside
    img_w, img_h = img_size
    surface_center_inside_img = (surface_centers[:, 0] >= 0) & (surface_centers[:, 1] >= 0) & \
                            (surface_centers[:, 0] <= img_w - 1) & (surface_centers[:, 1] <= img_h - 1)

    if surface_center_inside_img.sum() > 0:
        target_surface_center = surface_centers[surface_center_inside_img.argmax()]
        # y = ax + b
        a, b = np.polyfit([proj_center[0], target_surface_center[0]], [proj_center[1], target_surface_center[1]], 1)
        valid_intersects = []
        valid_edge = []

        left_y = b
        if (0 <= left_y <= img_h - 1):
            valid_intersects.append(np.array([0, left_y]))
            valid_edge.append(0)

        right_y = (img_w - 1) * a + b
        if (0 <= right_y <= img_h - 1):
            valid_intersects.append(np.array([img_w - 1, right_y]))
            valid_edge.append(1)

        top_x = -b / a
        if (0 <= top_x <= img_w - 1):
            valid_intersects.append(np.array([top_x, 0]))
            valid_edge.append(2)

        bottom_x = (img_h - 1 - b) / a
        if (0 <= bottom_x <= img_w - 1):
            valid_intersects.append(np.array([bottom_x, img_h - 1]))
            valid_edge.append(3)

        valid_intersects = np.stack(valid_intersects)
        min_idx = np.argmin(np.linalg.norm(valid_intersects - proj_center.reshape(1, 2), axis=1))
        
        return valid_intersects[min_idx], valid_edge[min_idx]
    else:
        return None


def convertRot2Alpha(ry3d, z3d, x3d):
    alpha = ry3d - math.atan2(x3d, z3d)

    # equivalent
    equ_alpha = ry3d - math.atan2(x3d, z3d)
    
    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha
