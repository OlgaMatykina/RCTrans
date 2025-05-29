import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pyquaternion import Quaternion


def get_3d_box(size, rotation, translation):
    w, l, h = size
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    y_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]

    corners = np.array([x_corners, y_corners, z_corners])  # (3, 8)

    # Ротируем каждую точку отдельно
    rotated_corners = np.array([rotation.rotate(pt) for pt in corners.T])  # (8, 3)

    translated_corners = rotated_corners + np.array(translation).reshape(1, 3)  # (8, 3)

    return translated_corners

def draw_box_on_image(img, corners, intrinsic):
    """Рисует линии между 2D-проекциями углов 3D бокса."""
    # Проецируем 3D точки в 2D
    points_3d = corners.T  # (3, 8)
    points_3d = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))  # (4, 8)
    pts_2d = intrinsic @ points_3d  # (3, 8)
    pts_2d = pts_2d[:2] / pts_2d[2]  # (2, 8)
    pts_2d = pts_2d.T.astype(np.int32)

    # Определим рёбра куба
    edges = [
        [0,1],[1,2],[2,3],[3,0],  # нижняя грань
        [4,5],[5,6],[6,7],[7,4],  # верхняя грань
        [0,4],[1,5],[2,6],[3,7]   # вертикали
    ]

    for start, end in edges:
        pt1 = tuple(pts_2d[start])
        pt2 = tuple(pts_2d[end])
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    return img

def visualize_predictions_on_images(pred_json_path, info_pkl_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(pred_json_path) as f:
        predictions = json.load(f)

    predictions = predictions['results']
    infos = pickle.load(open(info_pkl_path, 'rb'))
    # info_dict = {os.path.basename(i['image_path']).split('.')[0]: i for i in infos}

    for sample_token, preds in predictions.items():
        info = infos[int(sample_token)]
        assert info is not None

        img = cv2.imread(info['image_path'])
        intrinsic = np.array([
                    [1260.0, 0.0,   640.0, 0.],
                    [0.0,   1260.0, 360.0, 0.],
                    [0.0,   0.0,    1.0,   0.],
                    [0.0,   0.0,    0.0,   1.]
                ], dtype=np.float32)
        extrinsic = np.array([
                    [1., 0., 0., 0.6 ],
                    [ 0., 0.9961947 , -0.08715574, -0.3],
                    [ 0., 0.08715574, 0.9961947 , -0.4],
                    [ 0., 0.        , 0.        ,  1. ],
                ], dtype=np.float32)

        for pred in preds:
            size = pred['size']
            translation = pred['translation']
            rot = pred['rotation']  # [x, y, z, w]
            rotation = Quaternion([rot[3], rot[0], rot[1], rot[2]])  # [w, x, y, z]

            # Преобразуем 3D бокс из world в камеру
            box = Quaternion(pred['rotation'])
            corners = get_3d_box(size, rotation=rotation, translation=translation)
            corners = corners.T
            corners = np.vstack((corners, np.ones((1, corners.shape[1]))))
            corners_cam = extrinsic @ corners
            corners_cam = corners_cam[:3].T

            img = draw_box_on_image(img, corners_cam, intrinsic)

        out_path = os.path.join(output_dir, f"{sample_token}.png")
        cv2.imwrite(out_path, img)
        print(f"Saved: {out_path}")

# Использование:
visualize_predictions_on_images(
    pred_json_path='/home/docker_rctrans/test/rcdetr_90e_256×704_dino_vega/Fri_May_23_14_11_58_2025/pts_bbox/results_nusc.json',
    info_pkl_path='/home/docker_rctrans/HPR3/unpack_bags/info.pkl',
    output_dir='output_images'
)
