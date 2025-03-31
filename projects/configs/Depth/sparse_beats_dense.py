_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
randomness = dict(seed = 2024)
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_range = [-51.2, -51.2, 51.2, 51.2]
voxel_size = [0.2, 0.2, 8]
radar_voxel_size = [0.8, 0.8, 8]
voxel_size = [0.2, 0.2, 8]
# x y z rcs vx_comp vy_comp x_rms y_rms vx_rms vy_rms
radar_use_dims = [0, 1, 2, 8, 9, 18]
lidar_use_dims = [0, 1, 2, 3]
out_size_factor = 4
mem_query = 128

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# num_gpus = 8
num_gpus = 1
batch_size = 4
num_epochs = 90
num_iters_per_epoch = 323 // (num_gpus * batch_size)

queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'sensor2ego', 'ida_mat']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True)
model = dict(
    type='SBD_BIG',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint="ckpts/resnet18-nuimages-pretrained-e2e.pth",
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
        type='CPFPN',  ###remove unused parameters 
        in_channels=[256, 512],
        out_channels=256,
        num_outs=2),
    depth_model=dict(
        type='MatrixVT',
        x_bound=[-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        y_bound=[-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        z_bound=[-5, 3, 8],  # BEV grids bounds and size (m)
        d_bound=[2.0, 58.0, 0.5],  # Categorical Depth bounds and division (m)
        final_dim=(256, 704),  # img size for model input (pix)
        output_channels=64,  # BEV feature channels
        downsample_factor=16,  # ds factor of the feature to be projected to BEV (e.g. 256x704 -> 16x44)  # noqa
        depth_net_conf=dict(in_channels=256, mid_channels=256),
        loss_depth=dict(
            type='DepthLoss',
            dbound=[2.0, 58.0, 0.5],
            downsample_factor=16,
            loss_weight=3.0),
    ),
)


dataset_type = 'CustomNuScenesDepthDataset'
data_root = '../HPR3/nuscenes/'
ann_root = '../HPR3/'
file_client_args = dict(backend='disk')


ida_aug_conf = {
        "resize_lim": (0.38, 0.55),
        "final_dim": (256, 704),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewSegMaskFromFiles', semantic_mask_used_mask=[0, 1, 4, 12, 20, 32, 80, 83, 93, 127, 102, 116], to_float32=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=6,
        use_num=6,
        use_dim=radar_use_dims,
        max_num=2048),
    dict(
        type='LoadLidarPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=lidar_use_dims,
    ),
    dict(type='GenerateLidarDepth'),
    dict(type='GenerateRadarDepth'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='RadarRangeFilter', radar_range=bev_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True,
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='MyTransform',),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'radar', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists', 'lidar', 'depth_maps', 'radar_depth', 'seg_mask'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d','lidar2img','radar_aug_matrix', 'pcd_scale_factor'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=6,
        use_num=6,
        use_dim=radar_use_dims,
        max_num=2048),

    dict(
        type='LoadLidarPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=lidar_use_dims,
    ),
    dict(type='GenerateLidarDepth'),
    dict(type='GenerateRadarDepth'),

    dict(type='RadarRangeFilter', radar_range=bev_range),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False, with_depth=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32, training=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='MyTransform', training=True),
            dict(type='Collect3D', keys=['img','radar', 'lidar', 'depth_maps', 'radar_depth'] + collect_keys,
            meta_keys=('filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token','lidar2img'))
        ]), 
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_root + 'nuscenes_radar_temporal_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'radar', 'prev_exists', 'img_metas', 'depth_maps', 'radar_depth'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, data_root=data_root, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'radar', 'img_metas',], 
            queue_length=queue_length, ann_file=ann_root + 'nuscenes_radar_temporal_infos_val.pkl', classes=class_names, modality=input_modality, test_mode=True, seq_mode=True, seq_split_num=2),
    test=dict(type=dataset_type, data_root=data_root, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'radar', 'img_metas', ], 
            queue_length=queue_length, ann_file=ann_root + 'nuscenes_radar_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )

optimizer = dict(
    type='AdamW', 
    lr=1e-4, # bs 32 gpu 1 || bs 4 gpu 1: 1e-5 # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1), # set to 0.1 always better when apply 2D pretrained.
        }),
    weight_decay=0.01)

# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(type='GradientCumulativeFp16OptimizerHook', loss_scale='dynamic', cumulative_iters=8, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

# evaluation = dict(interval=1, pipeline=test_pipeline)
# evaluation = dict(interval=num_iters_per_epoch+1, pipeline=test_pipeline)
# evaluation = dict(interval=num_iters_per_epoch*num_epochs+1, pipeline=test_pipeline)
evaluation = dict(interval=1, metric='epe:0-80', save_best='epe:0-80', rule='less')


find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
# checkpoint_config = dict(interval=num_iters_per_epoch+1, max_keep_ckpts=3)
# checkpoint_config = dict(interval=1, max_keep_ckpts=3)
# runner = dict(
#     type='EpochBasedRunner', max_epochs=num_epochs)

checkpoint_config = dict(interval=1, max_keep_ckpts=3)

# checkpoint_config = dict(
#     max_keep_ckpts=3,
#     save_best='loss_val',  # Укажи метрику, по которой сохранять лучшие чекпоинты
#     rule='smaller'    # Большее значение лучше (обычно для mAP)
# )
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from=None
# resume_from='work_dirs/rctrans_gt_depth/iter_40004.pth'
resume_from=None
# custom_hooks = [dict(type='EMAHook')]
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='radar-camera',   # Название проекта в WandB
        #         name='lab_comp MatrixVT depth_net camera',     # Имя эксперимента
        #         config=dict(                # Дополнительные настройки эксперимента
        #             batch_size=batch_size,
        #             model='matrixvt',
        #         )
        #     )
        # ),
    ],
)

workflow = [('train', 1), ('val', 1)]