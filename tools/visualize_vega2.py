import os
import tqdm
import json
from visual_nuscenes_radar_cam_front_vega import Vega
use_gt = False
out_dir = '/home/docker_rctrans/RCTrans/result_vis/tmp2/'
result_json = "/home/docker_rctrans/test/rcdetr_90e_256×704_dino_vega/Sun_May_25_03_08_46_2025/pts_bbox/results_nusc"
dataroot='/home/docker_rctrans/HPR3/unpack_bags/'

os.makedirs(out_dir, exist_ok=True)


nusc = Vega(ann_file=dataroot+'info.pkl', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.05)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())
index=0
for token in tqdm.tqdm(tokens[:10]):
    index += 1
    if use_gt:
        nusc.render_sample(token, out_path = out_dir+str(index)+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = out_dir+str(index)+"_pred.png", verbose=False)

