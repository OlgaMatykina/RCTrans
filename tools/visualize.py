import os
import tqdm
import json
from visual_nuscenes_radar_cam_front import NuScenes
use_gt = False
out_dir = '/home/docker_rctrans/RCTrans/result_vis/tmp/'
result_json = "/home/docker_rctrans/test/rcdetr_90e_256×704_dino/Sun_May_25_03_14_49_2025/pts_bbox/results_nusc"
dataroot='/home/docker_rctrans/HPR3/nuscenes/'

os.makedirs(out_dir, exist_ok=True)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())
index=0
for token in tqdm.tqdm(tokens[10:15]):
    index += 1
    if use_gt:
        nusc.render_sample(token, out_path = out_dir+str(index)+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = out_dir+str(index)+"_pred.png", verbose=False)

