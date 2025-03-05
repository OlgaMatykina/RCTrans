# 3D object detection config file for nuScenes using PointPillars

_base_ = [
    '../../../mmdetection3d/configs/_base_/models/hv_pointpillars_fpn_nus.py',   # модель
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',      # настройки датасета
    '../../../mmdetection3d/configs/_base_/schedules/schedule_2x.py',  # настройка обучения
    '../../../mmdetection3d/configs/_base_/default_runtime.py',       # настройка работы
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Настройки датасета
dataset_type = 'NuScenesDataset'
data_root = '../HPR1/nuscenes/'
ann_root = '../HPR1/'

# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         file_client_args=file_client_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=10,
#         file_client_args=file_client_args),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-0.3925, 0.3925],
#         scale_ratio_range=[0.95, 1.05],
#         translation_std=[0, 0, 0]),
#     dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectNameFilter', classes=class_names),
#     dict(type='PointShuffle'),
#     dict(type='DefaultFormatBundle3D', class_names=class_names),
#     dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         file_client_args=file_client_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=10,
#         file_client_args=file_client_args),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(
#                 type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#             dict(
#                 type='DefaultFormatBundle3D',
#                 class_names=class_names,
#                 with_label=False),
#             dict(type='Collect3D', keys=['points'])
#         ])
# ]
# # construct a pipeline for data and gt loading in show function
# # please keep its loading function consistent with test_pipeline (e.g. client)
# eval_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=5,
#         file_client_args=file_client_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=10,
#         file_client_args=file_client_args),
#     dict(
#         type='DefaultFormatBundle3D',
#         class_names=class_names,
#         with_label=False),
#     dict(type='Collect3D', keys=['points'])
# ]

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=ann_root + 'mini_nuscenes_radar_temporal_infos_train.pkl',
        ),
    val=dict(
        data_root=data_root,
        ann_file=ann_root + 'mini_nuscenes_radar_temporal_infos_val.pkl',
        ),
    test=dict(
        data_root=data_root,
        ann_file=ann_root + 'mini_nuscenes_radar_temporal_infos_val.pkl',
    )
)

# Оптимизатор и параметры обучения
optimizer = dict(
    type='AdamW',
    lr=0.001,  # Начальная скорость обучения
    weight_decay=0.01,
)
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(
    policy='step',
    step=[16, 22],  # Шаги для уменьшения скорости обучения
    gamma=0.1,
)
total_epochs = 24  # Общее количество эпох

# Настройки для вывода
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='radar-camera',   # Название проекта в WandB
                name='my_first_model',     # Имя эксперимента
                config=dict(                # Дополнительные настройки эксперимента
                    batch_size=4,
                    learning_rate=0.001,
                    model='PointPillars',
                )
            )
        ),
    ],
)

# Параметры для оценки
evaluation = dict(interval=1, metric='mAP')

# Обучение с использованием multi-gpu
distributed = True
