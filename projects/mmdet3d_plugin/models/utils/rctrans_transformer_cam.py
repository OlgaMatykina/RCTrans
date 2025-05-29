# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import warnings
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import deprecated_api_warning, ConfigDict
import copy
from torch.nn import ModuleList
from .attention import FlashMHA
import torch.utils.checkpoint as cp
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.runner import auto_fp16

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RCTransTransformerDecoder_cam(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(RCTransTransformerDecoder_cam, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.bev_size = 128
        self.test_breaking = 2
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, key, value, key_pos, query_pos, temp_memory, temp_pos, key_padding_mask, attn_masks, \
                reg_branch, cls_branches, reg_branches, reference_points, img_metas, query_embed, temporal_alignment_pos):

        outputs_classes = []
        outputs_coords = []
        intermediate = []
        assert reference_points is not None
        
        # bev_key = key[:self.bev_size * self.bev_size, :, :]
        rv_key = key
        # bev_key_pos = key_pos[:self.bev_size * self.bev_size, :, :]
        rv_key_pos = key_pos

        # bev_temp_pos = temp_pos[0].transpose(1,0).contiguous()
        rv_temp_pos = temp_pos[0].transpose(1,0).contiguous()
        temp_memory = temp_memory.transpose(1,0).contiguous()

        # bev_query_pos = query_pos[0].transpose(1,0).contiguous()
        rv_query_pos = query_pos[0].transpose(1,0).contiguous()

        for index in range(int(len(self.layers)/2)):
            
            # print('query', query.shape, 'rv_key', rv_key.shape)
            query = self.layers[2*index](query, rv_key, rv_key, rv_query_pos, rv_key_pos, temp_memory, rv_temp_pos, attn_masks) # [Nq, B, C]
            query = self.layers[2*index + 1](query, rv_key, rv_key, rv_query_pos, rv_key_pos, temp_memory, rv_temp_pos, attn_masks) # [Nq, B, C]
            
            if self.post_norm is not None:
                temp_out = self.post_norm(query)
                
                temp_out = torch.nan_to_num(temp_out).transpose(1, 0)

                intermediate.append(temp_out)

                # predict
                reference = inverse_sigmoid(reference_points.clone())
                assert reference.shape[-1] == 3
                outputs_class =cls_branches[index](temp_out)
                tmp = reg_branches[index](temp_out)

                tmp[..., 0:3] += reference[..., 0:3]
                tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

                outputs_coord = tmp

                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            if index == self.test_breaking:
                if not self.training:
                    return outputs_classes, outputs_coords, torch.stack(intermediate)
            # update query pos
            if index < (int(len(self.layers)/2)-1):
                reference_points = tmp[..., 0:3].clone()
                rv_query_embeds = query_embed(reference_points, img_metas)
                rv_query_pos = temporal_alignment_pos(rv_query_embeds, reference_points)
                # bev_query_pos = bev_query_pos.transpose(1,0).contiguous()
                rv_query_pos = rv_query_pos.transpose(1,0).contiguous()

        return outputs_classes, outputs_coords, torch.stack(intermediate)