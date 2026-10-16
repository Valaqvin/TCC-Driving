#!/usr/bin/env python3
"""Smoke checks for the bargaining module integration path.

This script does not require CARLA. It verifies that RubinsteinBargaining
accepts both pre-encoded agent states, used by the CARLA wrapper, and BEV
feature maps, used by the original model path.
"""

import torch

from opencood.models.comm_modules.rubinstein_bargaining import RubinsteinBargaining


def run_case(name, spatial_features):
    torch.manual_seed(20260608)
    module = RubinsteinBargaining({
        'state_dim': 64,
        'offer_dim': 4,
        'hidden_dim': 128,
        'max_rounds': 1,
        'discount_alpha': 0.3,
        'waypoint_offset_scale': 2.0,
    })
    module.eval()

    n_agents = spatial_features.shape[0]
    waypoints = torch.zeros(1, 5, 2)
    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]])[:n_agents]
    velocities = torch.tensor([[5.0, 0.0], [-2.0, 0.0]])[:n_agents]
    headings = torch.zeros(n_agents)
    record_len = torch.tensor([n_agents])

    with torch.no_grad():
        out = module(
            spatial_features=spatial_features,
            waypoints=waypoints,
            positions=positions,
            velocities=velocities,
            headings=headings,
            record_len=record_len,
        )

    adjusted = out['adjusted_waypoints']
    delta = adjusted - waypoints
    meta = out['metadata'][0]
    assert adjusted.shape == waypoints.shape
    assert out['bargaining_offers'].shape[0] == n_agents
    assert float(delta.abs().max()) > 0.0
    assert meta['input_mode'] in {'encoded_state', 'bev_feature'}
    print(
        f'{name}: ok | input_mode={meta["input_mode"]} | '
        f'max_delta={float(delta.abs().max()):.4f} | '
        f'rounds={meta["n_rounds"]}')


def main():
    run_case('encoded_state_path', torch.randn(2, 64))
    run_case('bev_feature_path', torch.randn(2, 256, 8, 8))


if __name__ == '__main__':
    main()
