point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'CustomNuScenesDepthDataset'
data_root = '../HPR2/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=6,
        use_num=6,
        use_dim=[0, 1, 2, 8, 9, 18],
        max_num=2048),
    dict(
        type='LoadLidarPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3]),
    dict(type='GenerateLidarDepth'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True),
    dict(type='RadarRangeFilter', radar_range=[-51.2, -51.2, 51.2, 51.2]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.38, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='PETRFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego',
            'ida_mat', 'prev_exists'
        ]),
    dict(type='MyTransform'),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'radar', 'gt_bboxes',
            'gt_labels', 'centers2d', 'depths', 'prev_exists', 'lidar',
            'depth_maps', 'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego',
            'ida_mat'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'scene_token', 'gt_bboxes_3d',
                   'gt_labels_3d', 'lidar2img', 'radar_aug_matrix',
                   'pcd_scale_factor'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=6,
        use_num=6,
        use_dim=[0, 1, 2, 8, 9, 18],
        max_num=2048),
    dict(
        type='LoadLidarPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3]),
    dict(type='GenerateLidarDepth'),
    dict(type='RadarRangeFilter', radar_range=[-51.2, -51.2, 51.2, 51.2]),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.38, 0.55),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False,
        with_depth=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32, training=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego',
                    'ida_mat'
                ],
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='MyTransform', training=True),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'radar', 'lidar', 'depth_maps', 'lidar2img',
                    'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
                    'ego_pose', 'ego_pose_inv', 'sensor2ego', 'ida_mat'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token',
                           'lidar2img'))
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type='CustomNuScenesDepthDataset',
        data_root='../HPR2/nuscenes/',
        ann_file='../HPR2/mini_nuscenes_radar_temporal_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadRadarPointsMultiSweeps',
                load_dim=18,
                sweeps_num=6,
                use_num=6,
                use_dim=[0, 1, 2, 8, 9, 18],
                max_num=2048),
            dict(
                type='LoadLidarPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3]),
            dict(type='GenerateLidarDepth'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=True,
                with_label=True,
                with_bbox_depth=True),
            dict(
                type='RadarRangeFilter',
                radar_range=[-51.2, -51.2, 51.2, 51.2]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True),
            dict(
                type='GlobalRotScaleTransImage',
                rot_range=[-0.3925, 0.3925],
                translation_std=[0, 0, 0],
                scale_ratio_range=[0.95, 1.05],
                reverse_angle=True,
                training=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='PETRFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego',
                    'ida_mat', 'prev_exists'
                ]),
            dict(type='MyTransform'),
            dict(
                type='Collect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'radar',
                    'gt_bboxes', 'gt_labels', 'centers2d', 'depths',
                    'prev_exists', 'lidar', 'depth_maps', 'lidar2img',
                    'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
                    'ego_pose', 'ego_pose_inv', 'sensor2ego', 'ida_mat'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token',
                           'gt_bboxes_3d', 'gt_labels_3d', 'lidar2img',
                           'radar_aug_matrix', 'pcd_scale_factor'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=True,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        num_frame_losses=1,
        seq_split_num=2,
        seq_mode=True,
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego',
            'ida_mat', 'img', 'radar', 'prev_exists', 'img_metas', 'depth_maps'
        ],
        queue_length=1,
        use_valid_flag=True,
        filter_empty_gt=False),
    val=dict(
        type='CustomNuScenesDepthDataset',
        data_root='../HPR2/nuscenes/',
        ann_file='../HPR2/mini_nuscenes_radar_temporal_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadRadarPointsMultiSweeps',
                load_dim=18,
                sweeps_num=6,
                use_num=6,
                use_dim=[0, 1, 2, 8, 9, 18],
                max_num=2048),
            dict(
                type='LoadLidarPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3]),
            dict(type='GenerateLidarDepth'),
            dict(
                type='RadarRangeFilter',
                radar_range=[-51.2, -51.2, 51.2, 51.2]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False,
                with_depth=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32, training=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv', 'sensor2ego', 'ida_mat'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='MyTransform', training=True),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img', 'radar', 'lidar', 'depth_maps', 'lidar2img',
                            'intrinsics', 'extrinsics', 'timestamp',
                            'img_timestamp', 'ego_pose', 'ego_pose_inv',
                            'sensor2ego', 'ida_mat'
                        ],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'scene_token', 'lidar2img'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=True,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego',
            'ida_mat', 'img', 'radar', 'img_metas'
        ],
        queue_length=1,
        seq_mode=True,
        seq_split_num=2),
    test=dict(
        type='CustomNuScenesDepthDataset',
        data_root='../HPR2/nuscenes/',
        ann_file='../HPR2/mini_nuscenes_radar_temporal_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadRadarPointsMultiSweeps',
                load_dim=18,
                sweeps_num=6,
                use_num=6,
                use_dim=[0, 1, 2, 8, 9, 18],
                max_num=2048),
            dict(
                type='LoadLidarPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3]),
            dict(type='GenerateLidarDepth'),
            dict(
                type='RadarRangeFilter',
                radar_range=[-51.2, -51.2, 51.2, 51.2]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.38, 0.55),
                    final_dim=(256, 704),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False,
                with_depth=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32, training=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv', 'sensor2ego', 'ida_mat'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='MyTransform', training=True),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img', 'radar', 'lidar', 'depth_maps', 'lidar2img',
                            'intrinsics', 'extrinsics', 'timestamp',
                            'img_timestamp', 'ego_pose', 'ego_pose_inv',
                            'sensor2ego', 'ida_mat'
                        ],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'scene_token', 'lidar2img'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=True,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego',
            'ida_mat', 'img', 'radar', 'img_metas'
        ],
        queue_length=1),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ],
    metric='epe:0-80',
    save_best='epe:0-80',
    rule='less')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/matrixvt/'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
randomness = dict(seed=2024)
bev_range = [-51.2, -51.2, 51.2, 51.2]
voxel_size = [0.2, 0.2, 8]
radar_voxel_size = [0.8, 0.8, 8]
radar_use_dims = [0, 1, 2, 8, 9, 18]
lidar_use_dims = [0, 1, 2, 3]
out_size_factor = 4
mem_query = 128
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_gpus = 1
batch_size = 4
num_epochs = 90
num_iters_per_epoch = 80
queue_length = 1
num_frame_losses = 1
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
    'ego_pose', 'ego_pose_inv', 'sensor2ego', 'ida_mat'
]
model = dict(
    type='MatrixVT_BIG',
    num_frame_head_grads=1,
    num_frame_backbone_grads=1,
    num_frame_losses=1,
    use_grid_mask=True,
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/resnet18-nuimages-pretrained-e2e.pth',
            prefix='backbone.'),
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN', in_channels=[256, 512], out_channels=256, num_outs=2),
    depth_model=dict(
        type='MatrixVT',
        x_bound=[-51.2, 51.2, 0.8],
        y_bound=[-51.2, 51.2, 0.8],
        z_bound=[-5, 3, 8],
        d_bound=[2.0, 58.0, 0.5],
        final_dim=(256, 704),
        output_channels=64,
        downsample_factor=16,
        depth_net_conf=dict(in_channels=256, mid_channels=256),
        loss_depth=dict(
            type='DepthLoss',
            dbound=[2.0, 58.0, 0.5],
            downsample_factor=16,
            loss_weight=3.0)))
ann_root = '../HPR2/'
ida_aug_conf = dict(
    resize_lim=(0.38, 0.55),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    loss_scale='dynamic',
    cumulative_iters=8,
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
find_unused_parameters = False
runner = dict(type='IterBasedRunner', max_iters=7200)
gpu_ids = range(0, 1)
