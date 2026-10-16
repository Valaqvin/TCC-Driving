# Rubinstein Bargaining for Cooperative Driving
# ==============================================
#
# A differentiable Rubinstein alternating-offers bargaining protocol for
# multi-agent cooperative driving, integrated with the CoLMDriver framework.
#
# Core Idea
# ---------
# Instead of implicit feature-level negotiation (confidence maps + attention),
# vehicles engage in explicit strategic bargaining over trajectory adjustments.
# The Rubinstein (1982) model provides a unique subgame perfect equilibrium (SPE)
# solution: two players alternate offers, the "pie" shrinks proportionally to a
# discount factor over rounds, and the SPE is reached when the proposer's offer
# is accepted by the receiver.
#
# Key Innovation: TTC-Based Discount Factor
# -----------------------------------------
#    δ = 1 - exp(-α · TTC)
#
# This maps physical collision risk (Time-To-Collision) to economic impatience:
#   - TTC → ∞  ⇒  δ → 1  (patient, willing to bargain many rounds)
#   - TTC → 0  ⇒  δ → 0  (impatient, must accept quickly)
#
# Files Created
# -------------
# 1. opencood/models/comm_modules/rubinstein_bargaining.py
#    Core module with 5 components:
#      - compute_pairwise_ttc(): TTC matrix between all agent pairs
#      - TTCBasedDiscount: δ = 1 - exp(-α·TTC)
#      - UtilityEstimator: u(agent_state, offer) → ℝ  (differentiable)
#      - OfferDecoder: agent_state → initial_offer
#      - alternating_offers_protocol(): the bargaining game loop
#      - RubinsteinBargaining: main module orchestrating the above
#
# 2. opencood/models/center_point_codriving_with_bargaining.py
#    End-to-end CoLMDriver model variant that includes bargaining.
#    Extends center_point_codriving with an optional bargaining_module.
#
# 3. codriving/losses/bargaining.py
#    Bargaining-specific losses:
#      - BargainingLoss: composite (consistency + rationality + discount)
#      - BargainingConsistencyLoss: pairwise offer divergence
#      - BargainingRationalityLoss: individual rationality constraint
#
# Integration Architecture
# ------------------------
#
#   LiDAR Input
#       ↓
#   PillarVFE → Scatter → BEVBackbone
#       ↓
#   Where2comm (multi-scale attention fusion)
#       ↓
#   ┌──────────────────────────────────────┐
#   │  Rubinstein Bargaining (new)         │
#   │  1. Encode fused BEV → state vector  │
#   │  2. Compute pairwise TTC → δ         │
#   │  3. Generate initial offers          │
#   │  4. Alternating-offers protocol      │
#   │  5. Decode consensus → waypoint adj  │
#   └──────────────────────────────────────┘
#       ↓
#   Detection Heads + Adjusted Waypoints
#       ↓
#   V2X_Controller (steer/throttle/brake)
#
# Configuration Example
# ---------------------
# Add to your model config YAML:
#
#   model:
#     core_method: "center_point_codriving_with_bargaining"
#     args:
#       # ... existing args ...
#       bargaining:
#         enabled: true
#         state_dim: 64
#         offer_dim: 4
#         hidden_dim: 128
#         max_rounds: 3
#         discount_alpha: 0.3
#         delta_min: 0.05
#         delta_max: 0.99
#         waypoint_offset_scale: 2.0
#
# Loss Configuration Example
# --------------------------
#   loss:
#     args:
#       loss_configs:
#         - config:
#             type: BargainingLoss
#           weight: 1.0
#         - config:
#             type: WaypointL1Loss
#           weight: 1.0
#
# Training Strategy (3 Stages)
# -----------------------------
# Stage 1: Self-play pre-training
#   - Freeze perception backbone
#   - Train bargaining module in CARLA multi-agent scenarios
#   - Loss: BargainingConsistencyLoss + BargainingRationalityLoss
#
# Stage 2: Joint training
#   - Unfreeze planning heads
#   - Train bargaining + planning end-to-end
#   - Loss: BargainingLoss + WaypointL1Loss
#
# Stage 3: LLM guided fine-tuning
#   - Use LLM to annotate "socially optimal" offers from offline data
#   - Fine-tune UtilityEstimator to align with LLM judgments
#   - This bridges game-theoretic rationality with social norms
#
# Theoretical Justification
# -------------------------
# Rubinstein (1982): "Perfect Equilibrium in a Bargaining Model"
#   - Alternating offers + discounting → unique SPE
#   - SPE division: proposer gets (1-δ₂)/(1-δ₁δ₂)
#
# Scope of this release:
#   - Pairwise bargaining with a TTC-derived discount factor
#   - Differentiable pipeline for end-to-end learning
#   - Multi-vehicle coalition extensions are maintained outside this release
#
# Limitations & Future Work
# -------------------------
# - Current offer update rule is heuristic (moving toward mean of accepted offers).
#   Future: learn offer update via RL with the utility estimator as reward.
# - This release focuses on the pairwise bargaining core.
#   Multi-vehicle coalition research is maintained outside this release.
# - LLM integration is via offline annotation.
#   Future: online LLM as social norm regularizer during live bargaining.


