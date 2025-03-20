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
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from mmdet3d.models import builder
import torch.nn.functional as F
from torch.cuda.amp import autocast
from projects.mmdet3d_plugin import SPConvVoxelization

from projects.mmdet3d_plugin.models.backbones.matrixvt_slim import MatrixVT


@DETECTORS.register_module()
class RCDETR_MatrixVT(MVXTwoStageDetector):
    """RCDETR."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_dense_encoder=None,
                 radar_backbone=None,
                 radar_neck=None,
                 depth_loss=None,
                 depth_model=None,
                 ):
        super(RCDETR_MatrixVT, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False
        # radar encoder
        if radar_voxel_layer:
            self.radar_voxel_layer = SPConvVoxelization(**radar_voxel_layer)
        if radar_voxel_encoder:
            self.radar_voxel_encoder = builder.build_voxel_encoder(
                radar_voxel_encoder)
        if radar_middle_encoder:
            self.radar_middle_encoder = builder.build_middle_encoder(
                radar_middle_encoder)
        if radar_dense_encoder:
            self.radar_dense_encoder = builder.build_middle_encoder(
                radar_dense_encoder)
        else:
            self.radar_dense_encoder = None
        if radar_backbone:
            self.radar_backbone = builder.build_backbone(radar_backbone)
        else:
            self.radar_backbone = None
        if radar_neck is not None:
            self.radar_neck = builder.build_neck(radar_neck)
        else:
            self.radar_neck = None

        # if depth_loss is not None:
        #     self.depth_loss = build_loss(depth_loss)
        # else:
        #     self.depth_loss = None

        self.downsample_factor = 16
        self.dbound = [2.0, 58.0, 0.5]
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])

        # backbone_conf = {
        #     'x_bound': [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        #     'y_bound': [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        #     'z_bound': [-5, 3, 8],  # BEV grids bounds and size (m)
        #     'd_bound': [2.0, 58.0,
        #                 0.5],  # Categorical Depth bounds and division (m)
        #     'final_dim': (256, 704),  # img size for model input (pix)
        #     'output_channels':
        #     64,  # BEV feature channels
        #     'downsample_factor':
        #     16,  # ds factor of the feature to be projected to BEV (e.g. 256x704 -> 16x44)  # noqa
        #     'depth_net_conf':
        #     dict(in_channels=256, mid_channels=256),
        # }

        # self.depth_model = MatrixVT(**backbone_conf).cuda()
        self.depth_model = builder.build_backbone(depth_model)

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        # print('ORIGINAL IMAGE SHAPE', img.shape)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # print('IMAGE AFTER RESHAPE SHAPE', img.shape)

            img_feats = self.img_backbone(img)

            # print('AFTER RESNET18 IMAGE FEATURES SHAPE', type(img_feats), len(img_feats), img_feats[0].shape, img_feats[1].shape)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

            # print('AFTER IMG_NECK IMAGE FEATURES SHAPE', type(img_feats), len(img_feats), img_feats[0].shape, img_feats[1].shape)


        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)

        # print('AFTER ALL IMAGE FEATURES SHAPE', img_feats_reshaped.shape)
        # print('IDA MAT SHAPE', ida_mat.shape)
        # print('INTRINSICS SHAPE', intrinsics.shape)
        # print('SENSOR2EGO SHAPE', sensor2ego.shape)


        return img_feats_reshaped
    
    def img_feats_to_bev(self, img_feats, ida_mat, intrinsics, sensor2ego, gt_depth):
        if img_feats.dim() == 5:
            new_img_feats = img_feats.unsqueeze(dim=1)
            sensor2ego = sensor2ego.unsqueeze(dim=1)
            intrinsics = intrinsics.unsqueeze(dim=1)
            ida_mat = ida_mat.unsqueeze(dim=1)
        else:
            new_img_feats = img_feats

        # print('NEW_IMG_FEATS SHAPE', new_img_feats.shape)
        # print('BDA MAT SHAPE', torch.eye(4).unsqueeze(0).repeat(new_img_feats.shape[0], 1, 1).shape)
        # print('sensor2ego shape', sensor2ego.shape)
        # print('ida_mat shape', ida_mat.shape)
        # print('intrinsics shape', intrinsics.shape)

        
        # for inference and deployment where intrin & extrin mats are static
        # model.static_mat = model.get_proj_mat(mats_dict)

        bev_feature, depth = self.depth_model(
            new_img_feats, {
                'sensor2ego_mats': sensor2ego,
                'intrin_mats': intrinsics,
                'ida_mats': ida_mat,
                # 'sensor2sensor_mats': torch.rand((1, 1, 6, 4, 4)),
                'bda_mat': torch.eye(4).unsqueeze(0).repeat(new_img_feats.shape[0], 1, 1).to(intrinsics.device),
            },
            gt_depth,
            is_return_depth=True)

        # print('BEV SHAPE', bev_feature.shape, 'DEPTH SHAPE', depth.shape)

        return bev_feature, depth

    @force_fp32(apply_to=('radar'))
    def extract_radar_feat(self, radar):
        """Extract features of points."""
        voxels, num_points, coors = self.radar_voxelize(radar)
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)
        if self.radar_dense_encoder:
            x = self.radar_dense_encoder(x)
        if self.radar_backbone:
            x = self.radar_backbone(x)
            x = self.radar_neck(x)
            return x[0]
        else:
            return x
    
    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @auto_fp16(apply_to=('img', 'radar'), out_fp32=True)
    def extract_feat(self, img, radar, T, training_mode=False):
        """Extract features from images and points."""

        if radar is not None:
            img_feats = self.extract_img_feat(img, T, training_mode)
            radar_feats = self.extract_radar_feat(radar)
            return img_feats, radar_feats
        else:
            img_feats = self.extract_img_feat(img, T, training_mode)
            return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key not in ['radar', 'radar_feats', 'bev_feats', 'lidar', 'depth_maps']:
                    data_t[key] = data[key][:, i] 
                else:
                    data_t[key] = data[key]
            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)

            # depth_loss = self.loss_depth(data['depth_maps'], data['depth_preds'])
            depth_loss = self.depth_model.loss(data['depth_maps'], data['depth_preds'])
            losses['frame_'+str(i)+'_depth_loss'] = depth_loss
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
        return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)
        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi['topk_indexes']
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 

            return losses
        else:
            return None

    @force_fp32(apply_to=('img', 'radar'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # print('DATA KEYS', data.keys())
        # for key, value in data.items():
        #     if key != 'rescale' and key != 'img':
        #         try:
        #             print(key, len(value), len(value[0]), value[0][0].shape)
        #         except:
        #             continue
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        T = data['img'].size(1)
        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        # print('rec_img SHAPE', rec_img.shape)
        # print('prev_img SHAPE', prev_img.shape)

        gt_depths = data['depth_maps']
        # print('depth_maps', type(gt_depths), len(gt_depths), type(gt_depths[0]), gt_depths[0].shape)


        if isinstance(gt_depths, list):
            gt_depths = torch.stack(gt_depths, dim=1)

        # print('GT DEPTH SHAPE', gt_depths.shape)
        gt_depths = self.get_downsampled_gt_depth(gt_depths)
        # print('GT DEPTH DOWNSAMPLED SHAPE', gt_depths.shape)
    
        # prev_gt_depth = gt_depths[:, :-self.num_frame_backbone_grads]
        # gt_depth = gt_depths[:, -self.num_frame_backbone_grads:]
        # print('GT DEPTH SHAPE', gt_depth.shape)
        # print('prev_gt_depth SHAPE', prev_gt_depth.shape)
        # prev_gt_depth = self.get_downsampled_gt_depth(prev_gt_depth)

        ida_mat = data['ida_mat'][:, -self.num_frame_backbone_grads:]
        intrinsics = data['intrinsics'][:, -self.num_frame_backbone_grads:]
        sensor2ego = data['sensor2ego'][:, -self.num_frame_backbone_grads:]

        prev_ida_mat = data['ida_mat'][:, :-self.num_frame_backbone_grads]
        prev_intrinsics = data['intrinsics'][:, :-self.num_frame_backbone_grads]
        prev_sensor2ego = data['sensor2ego'][:, :-self.num_frame_backbone_grads]

        rec_radar = data['radar']
        rec_img_feats, rec_radar_feats = self.extract_feat(rec_img, rec_radar, self.num_frame_backbone_grads)
        bev_feats, depth = self.img_feats_to_bev(rec_img_feats, ida_mat, intrinsics, sensor2ego, gt_depths)
        
        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, None, T-self.num_frame_backbone_grads, True)
                prev_bev_feats, depth = self.img_feats_to_bev(prev_img_feats, prev_ida_mat, prev_intrinsics, prev_sensor2ego, gt_depths)

            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)

            # print('prev_bev_feats', prev_bev_feats.shape)
            # print('bev_feats', bev_feats.shape)
            data['bev_feats'] = torch.cat([prev_bev_feats, bev_feats], dim=1)
            # print('cat bev_feats', data['bev_feats'].shape)

            data['radar_feats'] = rec_radar_feats
        else:
            data['img_feats'] = rec_img_feats
            data['bev_feats'] = bev_feats
            # print('bev_feats', bev_feats.shape)
            data['radar_feats'] = rec_radar_feats
        
        data['depth_preds'] = depth

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)
        
        loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))

        # Debugging: Check for NaNs
        if torch.isnan(loss):
            print(f"🚨 Loss became NaN at iteration {self.iter}!")
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                    print(f"🚨 NaN detected in {k}: {v}")

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        if self.training:
            location = self.prepare_location(img_metas, **data)
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi['topk_indexes']
        else:
            location = None
            topk_indexes = None
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        # data['img_feats'] = self.extract_img_feat(data['img'], 1)
        # data['radar_feats'] = self.extract_radar_feat(data['radar'])
        ida_mat = data['ida_mat']
        intrinsics = data['intrinsics']
        sensor2ego = data['sensor2ego']

        gt_depths = data['depth_maps']
        # print('depth_maps', type(gt_depths), len(gt_depths), type(gt_depths[0]), gt_depths[0].shape)
        # print('img', type(data['img']), len(data['img']), type(data['img'][0]), data['img'][0].shape)

        if isinstance(gt_depths, list):
            gt_depths = torch.stack(gt_depths, dim=1)

        # print('GT DEPTH SHAPE', gt_depths.shape)
        if gt_depths.dim() == 4:
            gt_depths = gt_depths.unsqueeze(dim=1)
        gt_depths = self.get_downsampled_gt_depth(gt_depths)
        # print('GT DEPTH DOWNSAMPLED SHAPE', gt_depths.shape)

        rec_img_feats, rec_radar_feats = self.extract_feat(data['img'], data['radar'], 1)
        bev_feats, depth = self.img_feats_to_bev(rec_img_feats, ida_mat, intrinsics, sensor2ego, gt_depths)
        
        data['img_feats'] = rec_img_feats
        data['radar_feats'] = rec_radar_feats
        data['bev_feats'] = bev_feats

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N, d, H//downsample, W//downsample]
        """

        if isinstance(gt_depths, list):
            gt_depths = torch.stack(gt_depths, dim=0)

        B, S, N, H, W = gt_depths.shape
        gt_depths = gt_depths.contiguous().view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor, W // self.downsample_factor, 
                                self.downsample_factor * self.downsample_factor)

        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values

        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))

        # Теперь one_hot возвращает [B*N, H, W, d] вместо (B*N*h*w, d)
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.depth_channels + 1)[..., 1:]

        # Изменяем на [B*N, d, H, W]
        gt_depths = gt_depths.permute(0, 3, 1, 2).float()

        return gt_depths