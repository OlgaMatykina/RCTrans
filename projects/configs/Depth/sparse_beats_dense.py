_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
randomness = dict(seed = 2024)

num_gpus = 1
batch_size = 4
num_epochs = 30

model = dict(
    type='SBD'
)

dataset_type = 'Vidar'
data_root = '../HPR1/nuscenes/samples/'
semantic_root = '../HPR1/nuscenes/seg_mask/'
ann_root = '../HPR1/'
file_client_args = dict(backend='disk')

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        semantic_root=semantic_root,
        ann_file=ann_root + 'nuscenes_radar_temporal_infos_train.pkl',
        ),
    val=dict(type=dataset_type, data_root=data_root, semantic_root=semantic_root, ann_file=ann_root + 'nuscenes_radar_temporal_infos_val.pkl'),
    test=dict(type=dataset_type, data_root=data_root, semantic_root=semantic_root, ann_file=ann_root + 'nuscenes_radar_temporal_infos_val.pkl'),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )

optimizer = dict(
    type='AdamW', 
    lr=1e-3,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'img_backbone': dict(lr_mult=0.1), # set to 0.1 always better when apply 2D pretrained.
    #     }),
    weight_decay=0.01)

optimizer_config = dict(type='GradientCumulativeFp16OptimizerHook', loss_scale='dynamic', cumulative_iters=2, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

train_pipeline = [
    # здесь ваш пайплайн для тренировки
]

# evaluation = dict(interval=num_iters_per_epoch*num_epochs/4, pipeline=test_pipeline)
evaluation = dict(interval=1, pipeline=train_pipeline)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
# checkpoint_config = dict(interval=num_iters_per_epoch+1, max_keep_ckpts=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
runner = dict(
    type='EpochBasedRunner', max_iters=num_epochs)
load_from=None
resume_from=None
# custom_hooks = [dict(type='EMAHook')]
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='depth_estimation',   # Название проекта в WandB
                name='Sparse Beats Dense',     # Имя эксперимента
                config=dict(                # Дополнительные настройки эксперимента
                    batch_size=batch_size,
                    model='sbd',
                )
            )
        ),
    ],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    sampler=dict(type='DefaultSampler'),
)