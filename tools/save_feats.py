import sys
import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.abspath(os.path.join("/home/docker_rctrans/RCTrans")))

# from mmdet3d.registry import MODELS, DATASETS, TRANSFORMS
# from mmengine.runner import load_checkpoint
# from mmengine.model import revert_sync_batchnorm
# from mmengine.registry import build_from_cfg

# from mmcv import Config as MMCVConfig
# from mmcv.runner import build_optimizer
# from mmdet3d.utils import register_all_modules

# from mmengine.runner import set_random_seed

# ✅ твоя модель
from projects.mmdet3d_plugin.models.backbones.nets.dino_v2_with_adapter.dino_v2_adapter.dinov2_adapter import DinoAdapter
import time

import argparse


def main():
    parser = argparse.ArgumentParser(description='Save intermediate tensors')
    parser.add_argument('--part', help='train or val', default='train')
    args = parser.parse_args()

    cfg_path = '/home/docker_rctrans/RCTrans/projects/configs/RCTrans/dinov2.py'  # твой базовый конфиг
    cfg = Config.fromfile(cfg_path)

    # register_all_modules()

    # 🔧 Dataset
    if args.part == 'train':
        dataset = build_dataset(cfg.data.train)
    else:
        dataset = build_dataset(cfg.data.val)

    # 🔧 DataLoader
    dataloader = build_dataloader(
        dataset=dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        shuffle=False,
        drop_last=False
    )

    # ✅ Создаём модель
    model = DinoAdapter(add_vit_feature=False, pretrain_size=518, pretrained_vit=True,
                      num_heads=6, embed_dim=384, freeze_dino=True)
    model.cuda()
    model.eval()

    total_infer_time = 0.0
    num_batches = 0

    # 🔁 Прогон и сохранение
    if args.part == 'train':
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                img_metas = data['img_metas'].data[0][0][0]['filename']
                img = data['img'].data[0][0][0].squeeze(0).squeeze(0).cuda()
                # Прямой вызов твоей модели
                start_time = time.time()
                outs, x, c, cls = model.extract_intermediate_features(img)
                end_time = time.time()
                infer_time = end_time - start_time
                total_infer_time += infer_time
                num_batches += 1

                # сохраняем
                save_intermediate_tensors(batch_idx, img_metas, outs, x, c, cls)

            # if batch_idx > 100:  # ограничь кол-во для отладки
            #     break

        if num_batches > 0:
            avg_infer_time = total_infer_time / num_batches
            print(f'\nAverage inference time per batch: {avg_infer_time:.4f} sec')

    else:
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                img_metas = data['img_metas'][0].data[0][0]['filename']
                img = data['img'][0].data[0][0].squeeze(0).squeeze(0).cuda()
                # Прямой вызов твоей модели
                start_time = time.time()
                outs, x, c, cls = model.extract_intermediate_features(img)
                end_time = time.time()
                infer_time = end_time - start_time
                total_infer_time += infer_time
                num_batches += 1

                # сохраняем
                save_intermediate_tensors(batch_idx, img_metas, outs, x, c, cls)

            # if batch_idx > 100:  # ограничь кол-во для отладки
            #     break

        if num_batches > 0:
            avg_infer_time = total_infer_time / num_batches
            print(f'\nAverage inference time per batch: {avg_infer_time:.4f} sec')


def save_intermediate_tensors(batch_idx, img_metas, outs, x, c, cls, save_root='/home/docker_rctrans/RCTrans/debug_intermediate'):
    os.makedirs(save_root, exist_ok=True)

    bs = len(img_metas)
    for i in range(bs):
        meta = img_metas[i]
        filename = meta # или meta['ori_filename'] в зависимости от структуры

        if 'CAM_FRONT__' not in filename:
            continue  # пропускаем ненужные камеры

        img_name = os.path.splitext(os.path.basename(filename))[0]
        save_dir = os.path.join(save_root)
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            'outs': [o.cpu() for o in outs],
            'x': x.cpu(),
            'c': c.cpu(),
            'cls': cls.cpu()
        }, os.path.join(save_dir, f"{img_name}.pt"))

if __name__ == '__main__':
    main()
