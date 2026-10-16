# -*- coding: utf-8 -*-
# PnP_infer with Rubinstein Bargaining
# Inserts the alternating-offers bargaining protocol between planning and control.

import os
import copy
import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple

from collections import OrderedDict
from opencood.tools import inference_utils, train_utils
from team_code.pnp_infer_action_e2e import PnP_infer
from opencood.models.comm_modules.rubinstein_bargaining import (
    RubinsteinBargaining,
    compute_pairwise_ttc,
)


class PnP_infer_with_bargaining(PnP_infer):
    """
    Extends PnP_infer with Rubinstein bargaining between waypoint prediction and action.
    """
    
    def __init__(self, config=None, ego_vehicles_num=1, 
                 perception_model=None, planning_model=None,
                 perception_dataloader=None, device=None):
        super().__init__(
            config=config,
            ego_vehicles_num=ego_vehicles_num,
            perception_model=perception_model,
            planning_model=planning_model,
            perception_dataloader=perception_dataloader,
            device=device,
        )
        
        bargain_cfg = config.get('bargaining', {})
        self.bargaining_enabled = bargain_cfg.get('enabled', False)
        self.bargain_debug = bargain_cfg.get('debug', True)
        
        if self.bargaining_enabled:
            print('[Bargaining] Initializing Rubinstein bargaining module...')
            self.bargaining_module = RubinsteinBargaining(bargain_cfg)
            self.bargaining_module.to(device)
            self.bargaining_module.eval()
            
            self.bargaining_state_dim = bargain_cfg.get('state_dim', 64)
            self.feature_adapter = None
            self.feature_adapter_in_dim = None
            result_root = os.environ.get('RESULT_ROOT', os.getcwd())
            os.makedirs(result_root, exist_ok=True)
            self.bargain_log_path = os.path.join(result_root, 'bargain_log.npy')
            
            mr = bargain_cfg.get('max_rounds', 3)
            da = bargain_cfg.get('discount_alpha', 0.3)
            print(f'[Bargaining] Max rounds: {mr}, Discount alpha: {da}')

            # Load PPO-trained checkpoint if specified
            ckpt_path = bargain_cfg.get('checkpoint', '')
            if ckpt_path and os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    missing, unexpected = self.bargaining_module.load_state_dict(
                        ckpt['model'], strict=False)
                    loaded_keys = len(ckpt['model'])
                else:
                    new_sd = {}
                    for k, v in ckpt.items():
                        if k.startswith('mean_net.'):
                            new_sd['offer_decoder.decoder.' + k[len('mean_net.'):]] = v
                        elif k.startswith('decoder.'):
                            new_sd['offer_decoder.' + k] = v
                    missing, unexpected = self.bargaining_module.load_state_dict(
                        new_sd, strict=False)
                    loaded_keys = len(new_sd)
                print(f'[Bargaining] Loaded PPO checkpoint: {ckpt_path}')
                print(f'[Bargaining]   Parameters loaded: {loaded_keys} keys')
                if self.bargain_debug:
                    print(f'[Bargaining]   Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
            else:
                print(f'[Bargaining] No checkpoint at: {ckpt_path}')

            self.bargain_log = []
            self.bargain_failures = 0
        else:
            self.bargaining_module = None
            print('[Bargaining] Disabled.')
    
    def _extract_kinematics(self, car_data_raw):
        N = len(car_data_raw)
        device = self.device
        positions = torch.zeros(N, 2, device=device)
        velocities = torch.zeros(N, 2, device=device)
        headings = torch.zeros(N, device=device)
        
        for i in range(N):
            data = car_data_raw[i]
            if data is None:
                continue
            if 'measurements' in data:
                mes = data['measurements']
                positions[i, 0] = mes.get('lidar_pose_gps_x', 0)
                positions[i, 1] = mes.get('lidar_pose_gps_y', 0)
                speed_ms = mes.get('speed', 0.0)
                theta = mes.get('compass', 0.0)
                headings[i] = theta
                velocities[i, 0] = speed_ms * math.cos(theta)
                velocities[i, 1] = speed_ms * math.sin(theta)
        return {'positions': positions, 'velocities': velocities, 'headings': headings}
    
    def _get_fused_feature(self):
        if not hasattr(self, 'perception_memory_bank') or len(self.perception_memory_bank) == 0:
            return None
        latest = self.perception_memory_bank[-1]
        if 'feature' not in latest or latest['feature'] is None:
            return None
        return latest['feature']

    def _encode_bargaining_feature(self, fused_feat):
        if fused_feat.dim() == 2:
            if fused_feat.shape[-1] != self.bargaining_state_dim:
                raise ValueError(
                    f'Encoded bargaining feature dim mismatch: '
                    f'{fused_feat.shape[-1]} != {self.bargaining_state_dim}')
            return fused_feat
        if fused_feat.dim() != 4:
            raise ValueError(
                f'Expected fused feature as (N,C,H,W) or encoded (N,D), '
                f'got {tuple(fused_feat.shape)}')

        pooled = torch.nn.functional.adaptive_avg_pool2d(fused_feat, 1).flatten(1)
        in_dim = pooled.shape[-1]
        if self.feature_adapter is None or self.feature_adapter_in_dim != in_dim:
            self.feature_adapter_in_dim = in_dim
            self.feature_adapter = torch.nn.Sequential(
                torch.nn.Linear(in_dim, self.bargaining_state_dim),
                torch.nn.LayerNorm(self.bargaining_state_dim),
                torch.nn.ReLU(),
            ).to(self.device)
            self.feature_adapter.eval()
            print(
                f'[Bargaining] Feature adapter initialized: '
                f'{in_dim} -> {self.bargaining_state_dim}')
        return self.feature_adapter(pooled)
    
    def _log_bargaining(self, step, positions, velocities, offers, waypoints, adjusted_waypoints, metadata):
        n_agents = positions.shape[0]
        ttc_mat = compute_pairwise_ttc(positions, velocities, torch.zeros(n_agents, device=positions.device))
        meta = metadata[0] if isinstance(metadata, list) else metadata
        delta_wp = (adjusted_waypoints[0] - waypoints[0]).detach().cpu().numpy()[:2]
        finite_ttc = ttc_mat[torch.isfinite(ttc_mat)]
        min_ttc = float(finite_ttc.min().detach().cpu()) if finite_ttc.numel() else float('inf')
        max_delta = float(np.abs(delta_wp).max())
        log_entry = {
            'step': int(step),
            'n_agents': int(n_agents),
            'input_mode': meta.get('input_mode', 'unknown'),
            'n_rounds': int(meta.get('n_rounds', -1)),
            'converged': bool(meta.get('converged', False)),
            'min_ttc': min_ttc,
            'max_waypoint_delta': max_delta,
            'offers': offers.detach().cpu().numpy(),
            'waypoint_delta_first2': delta_wp,
        }
        self.bargain_log.append(log_entry)
        if len(self.bargain_log) == 1 or len(self.bargain_log) % 10 == 0:
            np.save(self.bargain_log_path, np.array(self.bargain_log, dtype=object))

        if self.bargain_debug:
            print(
                f'[Bargaining Step {step}] N={n_agents}, '
                f'Input={log_entry["input_mode"]}, Rounds={log_entry["n_rounds"]}, '
                f'Conv={log_entry["converged"]}, MinTTC={min_ttc:.3f}, '
                f'MaxWPDelta={max_delta:.3f}')
            for i in range(n_agents):
                off = offers[i].detach().cpu().numpy()
                print(f'  Agent {i}: [{off[0]:.3f},{off[1]:.3f},{off[2]:.3f},{off[3]:.3f}]')
            print(f'  WP d: [{delta_wp[0,0]:.3f},{delta_wp[0,1]:.3f}] [{delta_wp[1,0]:.3f},{delta_wp[1,1]:.3f}]')

    def _log_bargaining_skip(self, step, reason, extra=None):
        entry = {
            'step': int(step),
            'status': 'skipped',
            'reason': reason,
            'extra': extra or {},
        }
        self.bargain_log.append(entry)
        if len(self.bargain_log) == 1 or len(self.bargain_log) % 10 == 0:
            np.save(self.bargain_log_path, np.array(self.bargain_log, dtype=object))
        if self.bargain_debug:
            print(f'[Bargaining Step {step}] skipped: {reason} {extra or {}}')
    
    def get_action_from_list_multi_vehicle(self, car_data_raw, rsu_data_raw, step, timestamp):
        if not self.bargaining_enabled:
            return super().get_action_from_list_multi_vehicle(
                car_data_raw, rsu_data_raw, step, timestamp)
        
        # Perception + planning (adapted from parent)
        if 'comm_latency' in self.config['simulation']:
            raw_data_dict = {'car_data': car_data_raw, 'rsu_data': rsu_data_raw}
            self.pre_raw_data_bank.update({step: raw_data_dict})
            latency_step = self.config['simulation']['comm_latency']
            sorted_keys = sorted(list(self.pre_raw_data_bank.keys()))
            if step > latency_step:
                if step - sorted_keys[0] > latency_step:
                    self.pre_raw_data_bank.pop(sorted_keys[0])
                if step - latency_step in self.pre_raw_data_bank:
                    raw_data_used = self.pre_raw_data_bank[step - latency_step]
                    for i in range(len(car_data_raw)):
                        if i > 0:
                            car_data_raw[i] = raw_data_used['car_data'][i]
                    rsu_data_raw = raw_data_used['rsu_data']
        
        car_data, car_mask = self.check_data(car_data_raw)
        rsu_data, _ = self.check_data(rsu_data_raw, car=False)
        batch_data = self.collate_batch_infer_perception(car_data, rsu_data)
        extra_source = {'car_data': car_data_raw, 'rsu_data': rsu_data_raw}
        data = self.perception_dataloader.__getitem__(idx=None, extra_source=extra_source)
        batch_data_perception = [data]
        batch_data_perception = self.perception_dataloader.collate_batch_test(
            batch_data_perception, online_eval_only=True)
        batch_data_perception = train_utils.to_device(batch_data_perception, self.device)
        
        infer_result = inference_utils.inference_intermediate_fusion_multiclass(
            batch_data_perception, self.perception_model,
            self.perception_dataloader, online_eval_only=True)
        
        output_dict = OrderedDict()
        for cav_id, cav_content in batch_data_perception.items():
            output_dict[cav_id] = self.perception_model(cav_content)
        pred_box_tensor, pred_score, gt_box_tensor =             self.perception_dataloader.post_process_multiclass(
                batch_data_perception, output_dict, online_eval_only=True)
        infer_result = {
            'pred_box_tensor': pred_box_tensor,
            'pred_score': pred_score,
            'gt_box_tensor': gt_box_tensor,
        }
        if 'comm_rate' in output_dict['ego']:
            infer_result.update({'comm_rate': output_dict['ego']['comm_rate']})
        
        # Build memory bank & planning input
        occ_map, da, fused_feature_3 = (None, [], None)
        perception_results = output_dict['ego']
        if 'fused_feature' in perception_results:
            fused_feature_2 = perception_results['fused_feature'].permute(0, 1, 3, 2)
            fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
        elif 'features_before_fusion' in perception_results:
            fused_feature_3 = perception_results['features_before_fusion']
        feature = fused_feature_3[:, :, :192, :] if fused_feature_3 is not None else None
        
        # Memory update
        if not hasattr(self, 'perception_memory_bank'):
            self.perception_memory_bank = []
        self.perception_memory_bank.append({
            'occ_map': occ_map if occ_map is not None else torch.zeros(1,1,192,96, device=self.device),
            'drivable_area': torch.stack(da) if isinstance(da, list) and len(da) > 0 else torch.zeros(1,1,192,96, device=self.device),
            'detmap_pose': batch_data['detmap_pose'][:len(car_data_raw)],
            'target': batch_data['target'][:len(car_data_raw)],
            'feature': feature,
        })
        if len(self.perception_memory_bank) > 5:
            self.perception_memory_bank.pop(0)
        
        if len(self.perception_memory_bank) < 5:
            for _ in range(5 - len(self.perception_memory_bank)):
                entry = copy.deepcopy(self.perception_memory_bank[-1])
                if entry['feature'] is not None:
                    entry['feature'] = entry['feature'].detach().clone()
                self.perception_memory_bank.append(entry)
        
        planning_input = self.generate_planning_input()
        predicted_waypoints = self.planning_model(planning_input)
        predicted_waypoints = predicted_waypoints['future_waypoints']
        
        # === BARGAINING ===
        kinematics = self._extract_kinematics(car_data_raw)
        fused_feat = self._get_fused_feature()
        
        if fused_feat is None:
            self._log_bargaining_skip(
                step, 'missing_fused_feature',
                {'output_keys': list(output_dict.get('ego', {}).keys())})
        elif kinematics['positions'].shape[0] < 2:
            self._log_bargaining_skip(
                step, 'insufficient_agents',
                {'n_agents': int(kinematics['positions'].shape[0])})
        else:
            try:
                bargaining_feat = self._encode_bargaining_feature(fused_feat)
                with torch.no_grad():
                    barg_out = self.bargaining_module(
                        spatial_features=bargaining_feat,
                        waypoints=predicted_waypoints,
                        positions=kinematics['positions'],
                        velocities=kinematics['velocities'],
                        headings=kinematics['headings'],
                        record_len=torch.tensor([kinematics['positions'].shape[0]], device=self.device),
                    )
                if 'adjusted_waypoints' in barg_out:
                    self._log_bargaining(step, kinematics['positions'], kinematics['velocities'],
                        barg_out['bargaining_offers'], predicted_waypoints,
                        barg_out['adjusted_waypoints'], barg_out['metadata'])
                    delta = barg_out['adjusted_waypoints'] - predicted_waypoints
                    delta = torch.clamp(delta, -2.0, 2.0)
                    predicted_waypoints = predicted_waypoints + delta
            except Exception as e:
                self.bargain_failures += 1
                print(f'[Bargaining] Error: {e}')
                import traceback; traceback.print_exc()
        
        # Generate control
        control_all = self.generate_action_from_model_output(
            predicted_waypoints, car_data_raw, rsu_data_raw,
            car_data, rsu_data, batch_data, planning_input, car_mask, step, timestamp)
        return control_all
    
    def save_bargain_log(self, filepath='bargain_log.npy'):
        if hasattr(self, 'bargain_log') and self.bargain_log:
            target = filepath if filepath != 'bargain_log.npy' else self.bargain_log_path
            np.save(target, np.array(self.bargain_log, dtype=object))
            print(f'[Bargaining] Log saved to {target}')
