# -*- coding: utf-8 -*-
# Rubinstein Bargaining Module for Cooperative Driving
# 
# Implements a differentiable alternating-offers bargaining protocol
# between multiple autonomous vehicles, grounded in Rubinstein's (1982)
# strategic bargaining theory.
#
# Key innovations:
#   1. TTC-based discount factor: δ = 1 - exp(-α * TTC)
#      - Binds impatience to physical collision risk
#      - Creates a natural bridge between game theory and physics
#   2. Differentiable bargaining rounds in a learned embedding space
#   3. Utility estimation via neural network that learns social preferences
#
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


def compute_pairwise_ttc(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    headings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise Time-To-Collision between agents.
    
    Args:
        positions: (N, 2) agent positions in BEV coordinates
        velocities: (N, 2) agent velocity vectors (m/s)
        headings: (N,) agent heading angles (radians)
    
    Returns:
        ttc_matrix: (N, N) pairwise TTC, INF if no collision course
    """
    N = positions.shape[0]
    device = positions.device
    
    rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
    rel_vel = velocities.unsqueeze(1) - velocities.unsqueeze(0)
    dist = torch.norm(rel_pos, dim=-1)
    rel_pos_unit = F.normalize(rel_pos, dim=-1, eps=1e-8)
    # With rel_pos = p_i - p_j and rel_vel = v_i - v_j, agents are closing
    # when their relative velocity points opposite to the relative position.
    closing_speed = -torch.sum(rel_vel * rel_pos_unit, dim=-1)
    
    mask = closing_speed > 0.1
    ttc = torch.full_like(dist, float('inf'))
    ttc[mask] = dist[mask] / closing_speed[mask]
    ttc.fill_diagonal_(float('inf'))
    return ttc


class TTCBasedDiscount(nn.Module):
    """
    Compute the discount factor δ from Time-To-Collision.
    δ = 1 - exp(-α * TTC)
    Large TTC → δ → 1 (patient). Small TTC → δ → 0 (impatient).
    """
    def __init__(self, alpha: float = 0.3, delta_min: float = 0.05, delta_max: float = 0.99):
        super().__init__()
        self.alpha = alpha
        self.delta_min = delta_min
        self.delta_max = delta_max
    
    def forward(self, ttc: torch.Tensor) -> torch.Tensor:
        valid_ttc = torch.where(
            torch.isfinite(ttc), ttc, torch.full_like(ttc, 1e6))
        delta = 1.0 - torch.exp(-self.alpha * valid_ttc)
        delta = torch.clamp(delta, self.delta_min, self.delta_max)
        if delta.dim() == 2:
            delta.fill_diagonal_(1.0)
        return delta


class UtilityEstimator(nn.Module):
    """Estimates utility u(agent_state, offer) → ℝ."""
    def __init__(self, state_dim: int = 64, offer_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.offer_encoder = nn.Sequential(
            nn.Linear(offer_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, agent_state: torch.Tensor, offer: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_encoder(agent_state)
        offer_emb = self.offer_encoder(offer)
        joint = torch.cat([state_emb, offer_emb], dim=-1)
        return self.joint_encoder(joint)


class OfferDecoder(nn.Module):
    """Decodes agent state → bargaining offer in (0,1) via sigmoid.
    Offer dims: [speed_factor, lateral_shift, time_headway, yield_willingness]"""
    def __init__(self, state_dim: int = 64, offer_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, offer_dim),
        )
    
    def forward(self, agent_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(agent_state))


def alternating_offers_protocol(
    offers, discount_factors, utility_estimator, agent_states,
    max_rounds=3, return_all_offers=False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Alternating-offers bargaining. Each round a different proposer updates
    offer toward consensus. Acceptance threshold = δ^remaining · u(status_quo).
    Converges when non-proposers accept.
    """
    N = offers.shape[0]
    device = offers.device
    delta = discount_factors.diag() if discount_factors.dim() == 2 else discount_factors
    
    status_quo_utility = utility_estimator(agent_states, offers).squeeze(-1)
    current_offers = offers.clone()
    all_offers = [offers.clone()] if return_all_offers else []
    agreement = torch.zeros(N, dtype=torch.bool, device=device)
    converged = False
    
    for round_idx in range(max_rounds):
        proposer_idx = round_idx % N
        utilities = utility_estimator(agent_states, current_offers).squeeze(-1)
        remaining = max_rounds - round_idx
        threshold = (delta ** remaining) * status_quo_utility
        
        non_proposers = torch.ones(N, dtype=torch.bool, device=device)
        non_proposers[proposer_idx] = False
        
        accepted = utilities >= threshold
        if accepted[non_proposers].all():
            agreement = accepted
            converged = True
            break
        
        if proposer_idx < N:
            accepted_offers = current_offers[accepted]
            if len(accepted_offers) > 0:
                target = accepted_offers.mean(dim=0)
                step = 0.3 * (1.0 - delta[proposer_idx].item())
                current_offers[proposer_idx] = (
                    current_offers[proposer_idx] * (1 - step) + target * step)
        
        if return_all_offers:
            all_offers.append(current_offers.clone())
    
    if not converged:
        agreement = accepted if 'accepted' in locals() else agreement
    
    metadata = {'n_rounds': round_idx + 1, 'converged': converged, 'delta_values': delta.detach().cpu()}
    if return_all_offers:
        metadata['all_offers'] = torch.stack(all_offers)
    return current_offers, agreement, metadata


class RubinsteinBargaining(nn.Module):
    """
    Main differentiable Rubinstein bargaining module.
    
    Input: BEV features, waypoints, positions, velocities, headings, record_len
    Process: Encode → TTC → δ → Generate offers → Alternating-offers → Waypoint adj.
    Output: Adjusted waypoints + consensus weights + bargaining diagnostics
    
    Integrates into CoLMDriver pipeline between perception fusion and planning.
    """
    def __init__(self, args: dict):
        super().__init__()
        self.state_dim = args.get('state_dim', 64)
        self.offer_dim = args.get('offer_dim', 4)
        self.max_rounds = args.get('max_rounds', 3)
        
        self.discount = TTCBasedDiscount(
            alpha=args.get('discount_alpha', 0.3),
            delta_min=args.get('delta_min', 0.05),
            delta_max=args.get('delta_max', 0.99),
        )
        
        self.state_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.ReLU(),
        )
        
        self.offer_decoder = OfferDecoder(
            state_dim=self.state_dim, offer_dim=self.offer_dim,
            hidden_dim=args.get('hidden_dim', 128))
        self.utility_estimator = UtilityEstimator(
            state_dim=self.state_dim, offer_dim=self.offer_dim,
            hidden_dim=args.get('hidden_dim', 128))
        
        self.offer_to_waypoint = nn.Sequential(
            nn.Linear(self.offer_dim, 32), nn.ReLU(), nn.Linear(32, 4))
        self.waypoint_offset_scale = args.get('waypoint_offset_scale', 2.0)

    def encode_states(self, spatial_features: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Accept either BEV feature maps or pre-encoded agent states."""
        if spatial_features.dim() == 2:
            if spatial_features.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected encoded bargaining states with dim {self.state_dim}, "
                    f"got {spatial_features.shape[-1]}.")
            return spatial_features, 'encoded_state'
        if spatial_features.dim() == 4:
            return self.state_encoder(spatial_features), 'bev_feature'
        raise ValueError(
            "RubinsteinBargaining expects spatial_features as either "
            f"(N, {self.state_dim}) states or (N, C, H, W) BEV maps, "
            f"got shape {tuple(spatial_features.shape)}.")
    
    def forward(self, spatial_features, waypoints, positions, velocities, headings, record_len):
        B = record_len.shape[0]
        device = spatial_features.device
        bargaining_states, input_mode = self.encode_states(spatial_features)
        
        cumsum = torch.cumsum(record_len, dim=0)
        batch_splits = torch.tensor_split(
            torch.arange(spatial_features.shape[0], device=device), cumsum[:-1].cpu())
        
        adj_wps, all_off, all_acc, all_w, all_meta = [], [], [], [], []
        for b in range(B):
            n_agents = record_len[b].item()
            idx = batch_splits[b]
            st = bargaining_states[idx]
            pos = positions[idx] if positions.dim() == 2 else positions[b]
            vel = velocities[idx] if velocities.dim() == 2 else velocities[b]
            hdg = headings[idx] if headings.dim() == 1 else headings[b]
            
            ttc_mat = compute_pairwise_ttc(pos, vel, hdg)
            delta = self.discount(ttc_mat).min(dim=1)[0]
            init_off = self.offer_decoder(st)
            
            if n_agents >= 2:
                agreed_off, accept, meta = alternating_offers_protocol(
                    init_off, delta, self.utility_estimator, st, self.max_rounds)
            else:
                agreed_off, accept = init_off, torch.ones(1, dtype=torch.bool, device=device)
                meta = {'n_rounds': 0, 'converged': True}
            
            ego_wp = waypoints[b:b+1].clone()
            offset = self.offer_to_waypoint(agreed_off[0:1]) * self.waypoint_offset_scale
            ego_wp[0, 0, 0] += offset[0, 0]; ego_wp[0, 0, 1] += offset[0, 1]
            ego_wp[0, 1, 0] += offset[0, 2]; ego_wp[0, 1, 1] += offset[0, 3]
            
            adj_wps.append(ego_wp)
            all_off.append(agreed_off)
            all_acc.append(accept)
            with torch.no_grad():
                w = self.utility_estimator(st, agreed_off).mean().item()
            all_w.append(torch.tensor([[w]], device=device))
            meta['input_mode'] = input_mode
            meta['min_ttc'] = (
                torch.min(ttc_mat[torch.isfinite(ttc_mat)]).detach().cpu()
                if torch.isfinite(ttc_mat).any()
                else torch.tensor(float('inf'))
            )
            all_meta.append(meta)
        
        return {
            'adjusted_waypoints': torch.cat(adj_wps, dim=0),
            'bargaining_offers': torch.cat(all_off, dim=0),
            'acceptance_mask': torch.cat(all_acc, dim=0),
            'bargaining_weights': torch.cat(all_w, dim=0),
            'metadata': all_meta,
        }
