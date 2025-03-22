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
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
# num_iters_per_epoch = 81 // (num_gpus * batch_size)
num_epochs = 90

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
    type='RCDETR_MatrixVT',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    # depth_loss=dict(type='DepthLoss', 
    #                 depth_channels=64, 
    #                 downsample_factor=16,  
    #                 d_bound=[2.0, 58.0, 0.5],  # Categorical Depth bounds and division (m)
    #                 loss_weight=3.0,
    #                 ),
    # img encoder
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
    img_roi_head=dict(
        type='FocalHead',
        num_classes=10,
        in_channels=256,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
        assigner2d=dict(
            type='HungarianAssigner2D',
            cls_cost=dict(type='FocalLossCost', weight=2.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))
        ),
    # radar encoder
    radar_voxel_layer=dict(
        num_point_features=6,
        max_num_points=10, 
        voxel_size=radar_voxel_size, 
        max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range),
    radar_voxel_encoder=dict(
        type='RadarFeatureNet',
        in_channels=6,
        feat_channels=[32, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(
            type='BN1d',
            eps=1.0e-3,
            momentum=0.01)
    ),
    radar_middle_encoder=dict(
        type='PointPillarsScatter_futr3d',
        in_channels=64,
        output_shape=[128, 128],
    ),
    radar_dense_encoder=dict(
        type='Radar_dense_encoder_tf',
    ),

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

    # detect head
    pts_bbox_head=dict(
        type='RCTransBEVHead',
        num_classes=10,
        in_channels_img=256,
        in_channels_radar=64,
        num_query=900,
        memory_len= mem_query*4 , # 用来调节Memory Queue的长度
        topk_proposals=mem_query, # 每帧所要保存的query
        num_propagated=mem_query,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='RCTransTemporalTransformer',
            decoder=dict(
                type='RCTransTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=out_size_factor,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range),)))


dataset_type = 'CustomNuScenesDataset'
data_root = '../HPR1/nuScenes2d/nuscenes/'
ann_root = '../HPR1/nuScenes2d/'
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
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'radar', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists', 'lidar', 'depth_maps'] + collect_keys,
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
            dict(type='Collect3D', keys=['img','radar', 'lidar', 'depth_maps'] + collect_keys,
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
        collect_keys=collect_keys + ['img', 'radar', 'prev_exists', 'img_metas', 'depth_maps'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, data_root=data_root, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'radar', 'img_metas',], queue_length=queue_length, ann_file=ann_root + 'nuscenes_radar_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, data_root=data_root, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'radar', 'img_metas', ], queue_length=queue_length, ann_file=ann_root + 'nuscenes_radar_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )

optimizer = dict(
    type='AdamW', 
    lr=8e-5, # bs 32 gpu 1 || bs 4 gpu 1: 1e-5 # bs 8: 2e-4 || bs 16: 4e-4
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

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
# evaluation = dict(interval=num_iters_per_epoch+1, pipeline=test_pipeline)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
# checkpoint_config = dict(interval=num_iters_per_epoch+1, max_keep_ckpts=3)
checkpoint_config = dict(interval=10001, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from=None
resume_from=None
# custom_hooks = [dict(type='EMAHook')]
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='radar-camera',   # Название проекта в WandB
                name='cds2 RCTrans transformer layer bev+rv features and pos_embeds',     # Имя эксперимента
                config=dict(                # Дополнительные настройки эксперимента
                    batch_size=batch_size,
                    model='rcdetr',
                )
            )
        ),
    ],
)

'''
mAP: 0.4741
mATE: 0.5399
mASE: 0.2735
mAOE: 0.5566
mAVE: 0.2081
mAAE: 0.1899
NDS: 0.5602
'''