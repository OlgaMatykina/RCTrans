import os
import cv2
import numpy as np
import pickle
import scipy
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud

from mmdet.datasets import DATASETS

import functools
import time

from pyquaternion import Quaternion
from matplotlib import cm

def tensor_to_frame(output_dict):

    def brighten(inp_srgb):
        return np.clip((100/inp_srgb.mean())*inp_srgb,0,255).astype(np.uint8)

    frame_dict = {}
    for output_type, output_frame in output_dict.items():
        if output_frame is None:
            continue
        if isinstance(output_frame, torch.Tensor):
            output_frame = output_frame.cpu().numpy()
        elif isinstance(output_frame, np.ndarray):
            pass
        if output_type in [
            "img", 
        ]:
            img = output_frame[0,...].transpose((1,2,0))
            img = np.clip(img, 0, 255).astype(np.uint8)
            frame_dict[output_type] = img
    
        elif output_type in [
            'pred', 
            "label",
            "radar",
        ]:
            img = output_frame.squeeze()
            img = np.clip(img, 0, 80).astype(np.uint8)            
            frame_dict[output_type] = colorize_depth_map(img/80)
        elif output_type in [
            'pred_mask', 
            "label_mask",
            "radar_mask",
            "valid_label",
            "valid_label_mask",
        ]:
            img = output_frame.squeeze()
            frame_dict[output_type] = img

    return frame_dict


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """From vod.frame without rounding to int"""

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    # uvs = np.round(uvs).astype(np.int)

    return uvs


def map_pointcloud1_to_pointcloud2(
    lidar_points,
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
    
    points = lidar_points.points.transpose((1, 0))
    return points


def map_pointcloud_to_image(
    lidar_points,
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
    points = map_pointcloud1_to_pointcloud2(lidar_points, lidar_calibrated_sensor, lidar_ego_pose,
                                            cam_calibrated_sensor, cam_ego_pose, min_dist)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    uvs = project_3d_to_2d(points[:, :3], np.array(cam_calibrated_sensor['camera_intrinsic']))

    return points, np.concatenate((uvs, points[:, 2:3]), 1)


def canvas_filter(data, shape):
    return np.all((data >= 0) & (data < shape[1::-1]), 1)


def _scale_pts(data, out_shape, input_shape):
    data[:, :2] *= (np.array(out_shape[::-1]) / input_shape[1::-1])
    return data


def get_depth_map(data, shape, input_shape=None):
    if input_shape is not None:
        data = _scale_pts(data.copy(), shape, input_shape)

    depth = np.zeros(shape + (data.shape[1] - 2, ), dtype=np.float32)
    if np.any(data[:, :2].max(0) >= shape[1::-1]) or data[:, :2].min() < 0:
        inds = canvas_filter(data[:, :2], shape)
        data = data[inds]
    depth[data[:, 1].astype(int), data[:, 0].astype(int)] = data[:, 2:]
    return depth.squeeze()


def get_radar_vert_map(radar, out_shape, input_shape=None):
    if input_shape is not None:
        radar = _scale_pts(radar.copy(), out_shape, input_shape)

    radar_map = np.full(out_shape + (9, ), 10000, dtype=np.float32)
    radar_map[radar[:, 1].astype(int), radar[:, 0].astype(int)] = radar[:, 3:]
    radar_map = scipy.ndimage.minimum_filter1d(radar_map, 3, 1)
    radar_map[radar_map == 10000] = 0
    return radar_map


def get_radar_map(data, shape, input_shape=None):
    if input_shape is not None:
        data = _scale_pts(data.copy(), shape, input_shape)

    depth = np.zeros(shape + (data.shape[1] - 2, ), dtype=np.float32)
    if np.any(data[:, :2].max(0) >= shape[1::-1]) or data[:, :2].min() < 0:
        inds = canvas_filter(data[:, :2], shape)
        data = data[inds]
    depth[:, data[:, 0].astype(int)] = data[:, 2:]
    return depth.squeeze()


def extend_height(cam_depth, camera_intrinsic, origin_dims, h0=0.25, h1=1.5):    
    camera_intrinsic = camera_intrinsic 
    H, W = origin_dims
    def getRelSize(camera_intrinsic, d, w=0.5, h=1.5):
        v = (h*camera_intrinsic[0][0])/d
        u = (w*camera_intrinsic[1][1])/d    
        return u, v #int(u),int(v)
    ret = cam_depth.copy()
    for depth in cam_depth:
        x,y,d = depth 
        _,v1 = getRelSize(camera_intrinsic, d, 0, h1)
        _,v0 = getRelSize(camera_intrinsic, d, 0, h0)
        y_list = np.arange(start=max(y-v0,0),stop=min(y+v1,H),step=1)
        ptsnum_after_extend = len(y_list)
        x = np.stack((np.array([x]*ptsnum_after_extend),y_list,np.array([d]*ptsnum_after_extend)),axis=1)
        ret = np.concatenate((ret,x),axis=0)
    return ret


def colorize_depth_map(data, mask=None, norm=False):
    if mask is None:
        mask = data > 0
    elif mask.dtype != bool:
        mask = (mask > 0).astype(bool)

    # data = np.exp(-data / 72.136)
    if norm:
        min_val = data[mask].min()
        data = (data - min_val) / (data.max() - min_val)
    else:
        data = np.clip(data, 0, 1)

    data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    # data = cv2.applyColorMap(data, cv2.COLORMAP_JET)
    data = cm.jet(data)
    data = (data[..., :3] * 255).astype(np.uint8)

    mask = np.stack([mask] * 3, 2)
    data[~mask] = 0
    return data


def log_rate_limited(min_interval=1):
    def decorator(should_record):
        last = 0

        @functools.wraps(should_record)
        def wrapper(*args, **kwargs):
            nonlocal last
            if time.time() - last < min_interval:
                return False
            ret = should_record(*args, **kwargs)
            last = time.time()
            return ret

        return wrapper

    return decorator


class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {"epoch": self.epoch, "minibatch": self.minibatch, "step": self.step}

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict["epoch"]
        self.minibatch = clock_dict["minibatch"]
        self.step = clock_dict["step"]


class conf:
    input_h, input_w = 900, 1600
    max_depth = 80
    min_depth = 0


rng = np.random.default_rng()

@DATASETS.register_module()
class Vidar(torch.utils.data.Dataset):
    
    def __init__(self, data_root, semantic_root, ann_file, pipeline, classes, modality, test_mode, box_type_3d):

        self.path = ann_file
        self.data_root = data_root
        self.semantic_root = semantic_root

        self.CLASSES = classes

        with open(self.path, 'rb') as f:
            self.infos = pickle.loads(f.read())
        
        self.flag = np.array([0] * len(self.infos), dtype=np.uint8)
        self.radar_load_dim = 18 # self.radar_data_conf["radar_load_dim"]
        self.radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17] # [x y z] dyn_prop id [rcs vx vy vx_comp vy_comp] is_quality_valid ambig_state [x_rms y_rms] invalid_state pdh0 [vx_rms vy_rms] + [timestamp_diff]
        
        self.semantic_mask_used_mask = [0, 1, 4, 12, 20, 32, 80, 83, 93, 127, 102, 116] 
        # {"wall": 0, "building": 1, "sky": 2, "floor": 3, "tree": 4, "ceiling": 5, "road": 6, "bed ": 7, "windowpane": 8, "grass": 9, "cabinet": 10, "sidewalk": 11, "person": 12, "earth": 13, "door": 14, "table": 15,
        # "mountain": 16, "plant": 17, "curtain": 18, "chair": 19, "car": 20, "water": 21, "painting": 22, "sofa": 23, "shelf": 24, "house": 25, "sea": 26, "mirror": 27, "rug": 28, "field": 29, "armchair": 30, "seat": 31, 
        # "fence": 32, "desk": 33, "rock": 34, "wardrobe": 35, "lamp": 36, "bathtub": 37, "railing": 38, "cushion": 39, "base": 40, "box": 41, "column": 42, "signboard": 43, "chest of drawers": 44, "counter": 45, "sand": 46,
        # "sink": 47, "skyscraper": 48, "fireplace": 49, "refrigerator": 50, "grandstand": 51, "path": 52, "stairs": 53, "runway": 54, "case": 55, "pool table": 56, "pillow": 57, "screen door": 58, "stairway": 59, "river": 60,
        # "bridge": 61, "bookcase": 62, "blind": 63, "coffee table": 64, "toilet": 65, "flower": 66, "book": 67, "hill": 68, "bench": 69, "countertop": 70, "stove": 71, "palm": 72, "kitchen island": 73, "computer": 74, "swivel chair": 75,
        # "boat": 76, "bar": 77, "arcade machine": 78, "hovel": 79, "bus": 80, "towel": 81, "light": 82, "truck": 83, "tower": 84, "chandelier": 85, "awning": 86, "streetlight": 87, "booth": 88, "television receiver": 89, "airplane": 90, 
        # "dirt track": 91, "apparel": 92, "pole": 93, "land": 94, "bannister": 95, "escalator": 96, "ottoman": 97, "bottle": 98, "buffet": 99, "poster": 100, "stage": 101, "van": 102, "ship": 103, "fountain": 104, "conveyer belt": 105, 
        # "canopy": 106, "washer": 107, "plaything": 108, "swimming pool": 109, "stool": 110, "barrel": 111, "basket": 112, "waterfall": 113, "tent": 114, "bag": 115, "minibike": 116, "cradle": 117, "oven": 118, "ball": 119, "food": 120,
        # "step": 121, "tank": 122, "trade name": 123, "microwave": 124, "pot": 125, "animal": 126, "bicycle": 127, "lake": 128, "dishwasher": 129, "screen": 130, "blanket": 131, "sculpture": 132, "hood": 133, "sconce": 134, "vase": 135,
        # "traffic light": 136, "tray": 137, "ashcan": 138, "fan": 139, "pier": 140, "crt screen": 141, "plate": 142, "monitor": 143, "bulletin board": 144, "shower": 145, "radiator": 146, "glass": 147, "clock": 148, "flag": 149}
        
        self.RADAR_PTS_NUM = 200
        
        # Todo support multi-view Depth Completion
        # Now we follow the previous research, only use the front Camera and Radar
        self.radar_use_type = 'RADAR_FRONT'
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'

    def __len__(self):
            return len(self.infos)
        
    def get_params(self, data):
        params = dict()
        if 'calibrated_sensor' in data.keys():
            params['sensor2ego'] = data['calibrated_sensor']
        else:
            params['sensor2ego'] = dict()
            if 'lidar2ego_translation' in data.keys():
                params['sensor2ego']['translation'] = data['lidar2ego_translation']
                params['sensor2ego']['rotation'] = data['lidar2ego_rotation']
            else:
                params['sensor2ego']['translation'] = data['sensor2ego_translation']
                params['sensor2ego']['rotation'] = data['sensor2ego_rotation']
            if 'cam_intrinsic' in data.keys():
                params['sensor2ego']['camera_intrinsic'] = data['cam_intrinsic']
        
        if 'ego_pose' in data.keys():
            params['ego2global'] = data['ego_pose']
        else:
            params['ego2global'] = dict()
            params['ego2global']['translation'] = data['ego2global_translation']
            params['ego2global']['rotation'] = data['ego2global_rotation']
        
        return params

    def __getitem__(self, index):
        if isinstance(self.infos, list):
            data = self.infos[index]
        else:
            data = self.infos['infos'][index]

        # get cameras images only for front
        try:
            camera_infos = data['cam_infos'][self.camera_use_type]
        except:
            camera_infos = data['cams'][self.camera_use_type]
        camera_params = self.get_params(camera_infos)
        try:
            camera_filename = camera_infos['filename'].split('samples/')[-1]
        except:
            camera_filename = camera_infos['data_path'].split('samples/')[-1]

        img = cv2.imread(os.path.join(self.data_root, camera_filename))

        # get radars only for front
        try:
            radar_infos = data['radar_infos'][self.radar_use_type][0]
        except:
            radar_infos = data['radars'][self.radar_use_type][0]

        radar_params = self.get_params(radar_infos)
        path = radar_infos['data_path'].split('samples/')[-1]
        radar_obj = RadarPointCloud.from_file(os.path.join(self.data_root, path))
        radar_all = radar_obj.points.transpose(1,0)[:, self.radar_use_dims]
        radar = np.concatenate((radar_all[:, :3], np.ones([radar_all.shape[0], 1])), axis=1)
        
        # get lidar top
        try:
            lidar_infos = data['lidar_infos'][self.lidar_use_type]
        except:
            lidar_infos = data
        lidar_params = self.get_params(lidar_infos)

        try:
            path = lidar_infos['filename'].split('samples/')[-1]
        except:
            path = lidar_infos['lidar_path'].split('samples/')[-1]

        lidar_obj = LidarPointCloud.from_file(os.path.join(self.data_root, path))
        lidar = lidar_obj.points.transpose(1,0)[:, :3]
        lidar = np.concatenate((lidar, np.ones([lidar.shape[0], 1])), axis=1)
        
        # get semantic mask of images
        name = camera_filename.split('/')[-1].replace('.jpg', '.png')
        seg_mask_path = os.path.join(self.semantic_root, name)
        seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
        seg_mask_roi = list()
        for i in self.semantic_mask_used_mask:
            seg_mask_roi.append(np.where(seg_mask==i, 1, 0))
        seg_mask_roi = np.sum(np.stack(seg_mask_roi, axis=0), axis=0)
        
        # project lidar and radar to image coordinates
        lidar_pts, lidar = map_pointcloud_to_image(lidar, lidar_params['sensor2ego'], lidar_params['ego2global'],
                                        camera_params['sensor2ego'], camera_params['ego2global'])
        
        radar_pts, radar = map_pointcloud_to_image(radar, radar_params['sensor2ego'], radar_params['ego2global'],
                                        camera_params['sensor2ego'], camera_params['ego2global'])
        
        radar_pts = radar_pts[:, :3]
        valid_radar_pts_cnt = radar_pts.shape[0]
        if valid_radar_pts_cnt <= self.RADAR_PTS_NUM:
            padding_radar_pts = np.zeros((self.RADAR_PTS_NUM, 3), dtype=radar_pts.dtype)
            padding_radar_pts[:valid_radar_pts_cnt,:] = radar_pts
        else:
            random_idx = sorted(rng.choice(range(valid_radar_pts_cnt), size=(self.RADAR_PTS_NUM,), replace=False))
            padding_radar_pts = radar_pts[random_idx,:]
            
        inds = (lidar[:, 2] > conf.min_depth) & (lidar[:, 2] < conf.max_depth)
        lidar = lidar[inds]
        
        # Filter out the Lidar point cloud with overlapping near and far depth
        # uvs, depths = lidar[:, :2], lidar[:, -1]
        # tree = scipy.spatial.KDTree(uvs)
        # res = tree.query_ball_point(uvs, conf.query_radius)
        # filter_mask = np.array([
        #     (depths[i] - min(depths[inds])) / depths[i] > 0.1
        #     for i, inds in enumerate(res)])
        # lidar[filter_mask] = 0
        lidar = get_depth_map(lidar[:, :3], img.shape[:2])
        
        inds = canvas_filter(radar[:, :2], img.shape[:2])
        radar = radar[inds]
        radar = get_radar_map(radar[:, :3], img.shape[:2])
        
        img        = Image.fromarray(img[...,::-1])  # BGR->RGB
        lidar      = Image.fromarray(lidar.astype('float32'), mode='F')
        radar      = Image.fromarray(radar.astype('float32'), mode='F')
        seg_mask_roi   = torch.from_numpy(seg_mask_roi.astype('float32'))[None]
        
        # Aug
        try:
            img, lidar, radar, seg_mask_roi = augmention(img, lidar, radar, seg_mask_roi)
        except:
            pass
        
        lidar, radar = (np.array(d) for d in (lidar, radar))
        
        lidar_mask, radar_mask = (
            (d > 0).astype(np.uint8) for d in (lidar, radar))
        
        lidar, radar = (d[None] for d in (lidar, radar))
        lidar_mask, radar_mask = (
            d[None] for d in (lidar_mask, radar_mask))
        
        img = np.array(img)[...,::-1] # RGB -> BGR
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        return img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar, lidar_mask, seg_mask_roi


def augmention(img:Image, lidar:Image, radar:Image, seg_mask:torch.Tensor):
    width, height = img.size
    _scale = rng.uniform(1.0, 1.3) # resize scale > 1.0, no info loss
    scale  = int(height * _scale)
    degree = np.random.uniform(-5.0, 5.0)
    flip   = rng.uniform(0.0, 1.0)
    # Horizontal flip
    if flip > 0.5:
        img   = TF.hflip(img)
        lidar = TF.hflip(lidar)
        radar = TF.hflip(radar)
        seg_mask   = TF.hflip(seg_mask)
    
    # Color jitter
    brightness = rng.uniform(0.6, 1.4)
    contrast   = rng.uniform(0.6, 1.4)
    saturation = rng.uniform(0.6, 1.4)

    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    
    # Resize
    img        = TF.resize(img,   scale, interpolation=InterpolationMode.BICUBIC)
    lidar      = TF.resize(lidar, scale, interpolation=InterpolationMode.NEAREST)
    radar      = TF.resize(radar, scale, interpolation=InterpolationMode.NEAREST)
    seg_mask   = TF.resize(seg_mask, scale, interpolation=InterpolationMode.NEAREST)

    # Crop
    width, height = img.size
    ch, cw = conf.input_h, conf.input_w
    h_start = rng.integers(0, height - ch)
    w_start = rng.integers(0, width - cw)

    img          = TF.crop(img,   h_start, w_start, ch, cw)
    lidar        = TF.crop(lidar, h_start, w_start, ch, cw)
    radar        = TF.crop(radar, h_start, w_start, ch, cw)
    seg_mask     = TF.crop(seg_mask, h_start, w_start, ch, cw)

    img     = TF.gaussian_blur(img, kernel_size=3, )

    return img, lidar, radar, seg_mask


if __name__ == '__main__':
    from utils import colorize_depth_map
    dataset = Vidar()
    dataset = iter(dataset)
    img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar, lidar_mask, seg_mask_roi = next(dataset)
    radar = np.clip(radar, 0, 80).astype(np.uint8).squeeze()
    radar = colorize_depth_map(radar/80)
    cv2.imwrite('test_Radar.png', radar)


# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import random
import math
@DATASETS.register_module()
class NuScenesSBDDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            queue.append(example)
        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def union2one(self, queue):
        for key in self.collect_keys:
            if key not in ['img_metas', 'radar', 'lidar']:
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            elif key == 'img_metas':
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            radar_info=None if 'radars' not in info else info['radars']
        )

        if self.modality['use_camera']:
            image_paths = []
            semantic_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            sensor2ego_mats = []

            # lidar_depths = []

            for cam_type, cam_info in info['cams'].items():
                if cam_type == 'CAM_FRONT':
                    img_timestamp.append(cam_info['timestamp'] / 1e6)
                    image_paths.append(cam_info['data_path'])
                    name = camera_filename.split('/')[-1].replace('.jpg', '.png')
                    seg_mask_path = os.path.join(self.semantic_root, name)
                    semantic_paths.append(seg_mask_path)
                    # obtain lidar to image transformation matrix
                    cam2lidar_r = cam_info['sensor2lidar_rotation']
                    cam2lidar_t = cam_info['sensor2lidar_translation']
                    cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                    lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt)
                    intrinsics.append(viewpad)
                    extrinsics.append(lidar2cam_rt)
                    lidar2img_rts.append(lidar2img_rt)


                    # sweep sensor to sweep ego
                    sweepsensor2sweepego_r = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                    sweepsensor2sweepego_t = cam_info['sensor2ego_translation']
                    sweepsensor2sweepego_rt = convert_egopose_to_matrix_numpy(sweepsensor2sweepego_r, sweepsensor2sweepego_t)

                    # sweep ego to global
                    sweepego2global_r = Quaternion(cam_info['ego2global_rotation']).rotation_matrix
                    sweepego2global_t = cam_info['ego2global_translation']
                    sweepego2global_rt = convert_egopose_to_matrix_numpy(sweepego2global_r, sweepego2global_t)

                    # global sensor to cur ego
                    keyego2global_r = Quaternion(cam_info['ego2global_rotation']).rotation_matrix
                    keyego2global_t = cam_info['ego2global_translation']
                    keyego2global_rt = convert_egopose_to_matrix_numpy(keyego2global_r, keyego2global_t)
                    global2keyego_rt = invert_matrix_egopose_numpy(keyego2global_rt)

                    sweepsensor2keyego = global2keyego_rt @ sweepego2global_rt @\
                        sweepsensor2sweepego_rt
                    sensor2ego_mats.append(sweepsensor2keyego)

                # point_depth = get_lidar_depth(
                #         sweep_lidar_points[sweep_idx], img,
                #         info, cam_info)
                # lidar_depths.append(point_depth)
                
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    semantic_filename=semantic_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                    sensor2ego=sensor2ego_mats,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'],
                    # lidar_depth=lidar_depths
                    )
            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix
