import sys
sys.path.append('/home/docker_rctrans/RCTrans')

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines import Compose
from projects.mmdet3d_plugin.datasets.pipelines import LoadRadarPointsMultiSweeps

file_client_args = dict(backend='disk')
radar_use_dims = [0, 1, 2, 8, 9, 18]

# Получи пайплайн
pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=6,
        use_num=6,
        use_dim=radar_use_dims,
        max_num=2048),
    ]

# Построй датасет
dataset_type = 'CustomNuScenesDataset'
data_root = '/home/docker_rctrans/HPR1/nuScenes2d/nuscenes/'
ann_root = '/home/docker_rctrans/HPR1/nuScenes2d/'
queue_length = 1
num_frame_losses = 1
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True)
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

dataset = build_dataset(dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_root + 'nuscenes_radar_temporal_infos_val.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        pipeline=pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'radar', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
        )

# Получи одно сырое аннотированное значение
raw_data = dataset.get_data_info(1)
print(len(raw_data['sweeps']))

# Прогони пайплайн
processed = Compose(pipeline)(raw_data)

# Выведи результат
for key in processed:
    print(key, type(processed[key]), getattr(processed[key], 'shape', ''))
