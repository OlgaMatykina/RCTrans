import torch
from collections import OrderedDict

# Загружаем старый state_dict
state_dict = torch.load("../ckpts/res18.pth")  # замените на свой файл

new_state_dict = OrderedDict()

for key, value in state_dict['state_dict'].items():
    # Переносим веса выходного проекционного слоя без изменений
    if 'attentions.1.attn' in key:
        new_state_dict[key.replace('attn', 'attn.inner_attn.attn')] = value
    new_state_dict[key] = value
# Сохраняем новый state_dict
state_dict['state_dict'] = new_state_dict
torch.save(state_dict, "../ckpts/res18_converted.pth")


