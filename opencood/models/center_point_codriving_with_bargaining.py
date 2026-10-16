# -*- coding: utf-8 -*-
# Author: Codex
# License: TDG-Attribution-NonCommercial-NoDistrib
#
# CenterPoint CoDriving model with Rubinstein Bargaining integration.
# 
# This extends center_point_codriving by adding an optional Rubinstein bargaining
# module that adjusts predicted waypoints through a differentiable
# alternating-offers protocol before they are passed to the controller.
#
# Key changes from base center_point_codriving:
# - Optional bargaining_module that operates on fused spatial features
# - Bargaining adjusts the first 2 waypoints based on TTC-discounted negotiation
# - Output dict includes bargaining diagnostics for training/debugging

import torch
import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.codriving_attn import Where2comm
from opencood.models.comm_modules.rubinstein_bargaining import RubinsteinBargaining


class CenterPointCoDrivingWithBargaining(nn.Module):
    """
    CoLMDriver perception + planning with Rubinstein bargaining.
    
    Architecture:
        LiDAR → PillarVFE → Scatter → BEVBackbone → Where2comm (fusion)
        → Detection heads → [Bargaining on fused feature] → adjusted waypoints
    
    The bargaining module operates on the fused BEV features and current
    waypoints, running an alternating-offers protocol between vehicles
    to produce consensus-aware trajectory adjustments.
    """
    
    def __init__(self, args):
        super(CenterPointCoDrivingWithBargaining, self).__init__()
        
        # ---- Perception backbone (same as center_point_codriving) ----
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range = args['lidar_range']
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        
        # Shrink header
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        
        # Compression
        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])
        
        # Where2comm fusion
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']
        
        # Detection heads
        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 8 * args['anchor_number'], kernel_size=1)
        
        # ---- Rubinstein Bargaining module (new) ----
        self.use_bargaining = args.get('bargaining', {}).get('enabled', False)
        if self.use_bargaining:
            # The bargaining module takes fused spatial features and waypoints
            # and produces consensus-adjusted trajectories
            self.bargaining_module = RubinsteinBargaining(args['bargaining'])
            
            # Optional: perception-to-bargaining feature adapter
            # This bridges the gap between detection features and bargaining state
            self.bargaining_feature_adapter = nn.Sequential(
                nn.Conv2d(self.out_channel, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        else:
            self.bargaining_module = None
        
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        
        if 'early_fusion' in args:
            self.early_flag = args['early_fusion']
        else:
            self.early_flag = False
        
        self.init_weight()
    
    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)
    
    def backbone_fix(self):
        """Fix parameters during fine-tuning."""
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    def forward(self, data_dict, waypoints=None, positions=None, velocities=None, headings=None):
        """
        Args:
            data_dict: dict with keys 'processed_lidar', 'record_len', 'pairwise_t_matrix'
            waypoints: (B, 10, 2) or None — current waypoint predictions
            positions: (sum_ncav, 2) or None — agent positions in BEV coords
            velocities: (sum_ncav, 2) or None — agent velocity vectors
            headings: (sum_ncav,) or None — agent heading angles
            
        Returns:
            output_dict with all standard perception outputs PLUS:
            - adjusted_waypoints: (B, 10, 2) bargaining-constrained waypoints
            - bargaining_offers: (sum_ncav, offer_dim) final offers
            - acceptance_mask: (sum_ncav,) binary acceptance indicators
            - bargaining_weights: (B, 1) consensus weights
        """
        # ---- 1. Perception backbone ----
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        
        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        # ---- 2. Single-vehicle detection head ----
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        
        # ---- 3. Where2comm multi-agent fusion ----
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                batch_dict['spatial_features'], psm_single,
                record_len, pairwise_t_matrix, self.backbone, waypoints)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        elif self.early_flag:
            fused_feature_tuple = self.regroup(spatial_features_2d, record_len)
            feature_bank = [f[0] for f in fused_feature_tuple]
            fused_feature = torch.stack(feature_bank, dim=0)
            result_dict = {}
            communication_rates = 0
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                spatial_features_2d, psm_single, record_len, pairwise_t_matrix)
        
        # ---- 4. Detection output ----
        cls = self.cls_head(fused_feature)
        bbox = self.reg_head(fused_feature)
        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)
        
        output_dict = {
            'cls_preds': cls,
            'reg_preds': bbox_temp,
            'bbox_preds': bbox,
            'comm_rate': communication_rates,
        }
        result_dict.update({'fused_feature': fused_feature})
        output_dict.update(result_dict)
        
        # ---- 5. Single-vehicle outputs ----
        _, bbox_temp_single = self.generate_predicted_boxes(psm_single, rm_single)
        output_dict.update({
            'cls_preds_single': psm_single,
            'reg_preds_single': bbox_temp_single,
            'bbox_preds_single': rm_single,
        })
        
        # ---- 6. Rubinstein Bargaining (new) ----
        if self.use_bargaining and waypoints is not None and positions is not None:
            bf = self.bargaining_feature_adapter(fused_feature)
            bargaining_output = self.bargaining_module(
                spatial_features=bf,
                waypoints=waypoints,
                positions=positions,
                velocities=velocities if velocities is not None else torch.zeros_like(positions),
                headings=headings if headings is not None else torch.zeros(positions.shape[0], device=positions.device),
                record_len=record_len,
            )
            
            output_dict['adjusted_waypoints'] = bargaining_output['adjusted_waypoints']
            output_dict['bargaining_offers'] = bargaining_output['bargaining_offers']
            output_dict['acceptance_mask'] = bargaining_output['acceptance_mask']
            output_dict['bargaining_weights'] = bargaining_output['bargaining_weights']
            output_dict['bargaining_metadata'] = bargaining_output['metadata']
        
        return output_dict
    
    def generate_predicted_boxes(self, cls_preds, box_preds):
        """Convert head outputs to bounding box predictions."""
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        batch, H, W, code_size = box_preds.size()
        box_preds = box_preds.reshape(batch, H * W, code_size)
        
        batch_reg = box_preds[..., 0:2]
        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h, w, l], dim=-1)
        batch_hei = box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2] + self.cav_lidar_range[2]
        
        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]
        rot = torch.atan2(batch_rots, batch_rotc)
        
        ys, xs = torch.meshgrid([
            torch.arange(0, H, device=cls_preds.device),
            torch.arange(0, W, device=cls_preds.device)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1)
        xs = xs.view(1, H, W).repeat(batch, 1, 1)
        
        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]
        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]
        
        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        return cls_preds, batch_box_preds
