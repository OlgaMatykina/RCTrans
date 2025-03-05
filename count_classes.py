import mmcv
from collections import Counter

# Загрузи pkl-файл с аннотациями
info_path = "../HPR1/nuscenes_radar_temporal_infos_val.pkl"
infos = mmcv.load(info_path)['infos']

# Подсчет количества объектов
class_counts = Counter()
for info in infos:
    for ann in info["gt_names"]:
        class_counts[ann] += 1

# Выведем статистику
for category, count in class_counts.items():
    print(f"{category}: {count}")