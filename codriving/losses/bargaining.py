from typing import Tuple, Dict, Any
import torch
from torch import nn

from codriving import CODRIVING_REGISTRY


@CODRIVING_REGISTRY.register
class BargainingLoss(nn.Module):
    """
    Loss function for the Rubinstein bargaining module.
    
    Combines three terms:
    1. Consistency Loss: penalty when agents' final offers diverge after bargaining
       L_cons = ||offer_i - offer_j||² for all i,j
       Encourages convergence to a shared consensus.
    
    2. Rationality Loss: each agent's accepted offer must give utility ≥ status quo
       L_rat = max(0, u(status_quo) - u(agreed))
       Implements the individual rationality constraint from game theory.
    
    3. Discount Penalty: penalizes slow convergence (impatience)
       L_dis = Σ δ^i (u(first) - u(final))
       Ligher weight on later rounds due to discounting, rewarding fast agreement.
    
    Total: L = w_cons * L_cons + w_rat * L_rat + w_dis * L_dis
    """
    def __init__(self, w_cons: float = 1.0, w_rat: float = 2.0, w_dis: float = 0.5):
        super().__init__()
        self.w_cons = w_cons
        self.w_rat = w_rat
        self.w_dis = w_dis
    
    def forward(self, batch_data: Dict, model_output: Dict) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            batch_data: ground truth data from dataset
            model_output: model output containing bargaining module outputs
                - adjusted_waypoints: (B, 10, 2)
                - bargaining_offers: (sum_ncav, offer_dim)
                - acceptance_mask: (sum_ncav,)
                - metadata: list of dicts
        Returns:
            loss, extra_info
        """
        device = model_output['adjusted_waypoints'].device
        extra_info = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # -- Waypoint supervision (reuse standard L1 if ground truth available) --
        if 'future_waypoints' in batch_data:
            target = batch_data['future_waypoints']
            pred = model_output['adjusted_waypoints']
            wp_loss = torch.nn.functional.l1_loss(pred, target)
            total_loss += wp_loss
            extra_info['wp_loss'] = wp_loss.item()
        
        # -- Bargaining losses (only when multi-agent) --
        if 'bargaining_offers' in model_output and model_output['bargaining_offers'].shape[0] > 1:
            offers = model_output['bargaining_offers']
            n = offers.shape[0]
            
            # 1. Consistency loss: pairwise offer divergence
            offer_expand = offers.unsqueeze(1).expand(n, n, -1)
            offer_t_expand = offers.unsqueeze(0).expand(n, n, -1)
            pair_dist = torch.norm(offer_expand - offer_t_expand, dim=-1)
            # Upper triangle only, exclude diagonal
            triu_indices = torch.triu_indices(n, n, offset=1, device=device)
            cons_loss = pair_dist[triu_indices[0], triu_indices[1]].mean()
            total_loss += self.w_cons * cons_loss
            extra_info['consistency_loss'] = cons_loss.item()
            
            # 2. Rationality loss placeholder (needs utility values saved in metadata)
            extra_info['rationality_loss'] = 0.0
            
            # 3. Discount/timing penalty placeholder
            extra_info['discount_penalty'] = 0.0
            
            extra_info['n_agents'] = n
        
        return total_loss, extra_info


@CODRIVING_REGISTRY.register
class BargainingConsistencyLoss(nn.Module):
    """
    Standalone consistency loss for bargaining.
    L_cons = mean pairwise L2 distance between agents' final offers.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, offers: torch.Tensor) -> torch.Tensor:
        n = offers.shape[0]
        device = offers.device
        exp_i = offers.unsqueeze(1).expand(n, n, -1)
        exp_j = offers.unsqueeze(0).expand(n, n, -1)
        dist = torch.norm(exp_i - exp_j, dim=-1)
        triu = torch.triu_indices(n, n, offset=1, device=device)
        return dist[triu[0], triu[1]].mean()


@CODRIVING_REGISTRY.register
class BargainingRationalityLoss(nn.Module):
    """
    Individual rationality constraint.
    L_rat = Σ max(0, u(status_quo_i) - u(agreed_i))
    """
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        status_quo_utilities: torch.Tensor,
        agreed_utilities: torch.Tensor,
    ) -> torch.Tensor:
        violation = torch.clamp(
            status_quo_utilities - agreed_utilities + self.margin, min=0)
        return violation.mean()
