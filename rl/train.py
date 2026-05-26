"""
train.py  —  3-stage curriculum DQN for Chain Reaction.

Usage
-----
    python rl/train.py
    python rl/train.py --episodes 8000
    python rl/train.py --resume rl/checkpoints/ep02000.pt

Curriculum
----------
  Stage 1  vs random          graduate: win rate >= GRAD_RANDOM    x 2 evals
  Stage 2  vs defensive       graduate: win rate >= GRAD_DEFENSIVE x 2 evals
  Stage 3  vs greedy_capture  graduate: win rate >= GRAD_GREEDY    x 2 evals
  After Stage 3 graduation, continues vs greedy for remaining episode budget.

Bellman target (1-step zero-sum)
---------------------------------
    target = r + gamma * (1 - done) * (-max Q_hat(s'))
s' is always from the opponent's perspective — negate their best value.

Exposure handling
-----------------
  Type 1 — fires in compute_shaping after your move if you created/kept exposure.
           Simulates each threatening enemy primed cell firing; penalises per orb lost.
  Type 2 — fires in runner when exposure existed at START of your turn.
           Force greedy action when a big cluster (>= 4 mixed primed cells) is present;
           reward REWARD_EXPOSURE_GAIN per enemy primed destroyed.

L-overload execution
--------------------
  Config: 3 my primed cells in a 2x2 block, 4th cell (any owner) at >= cm-2,
          4th cell adjacent to at least one enemy primed cell.
  When this config exists pre-move and enemy primed cells are destroyed post-cascade,
  REWARD_L_OVERLOAD_EXECUTE fires per enemy primed destroyed.
  We reward execution (not setup) to avoid incentivising half-built configs that
  leave the 4th cell exposed for the enemy to cascade into.
"""

import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.env           import ChainReactionEnv
from rl.agent         import DQNAgent
from rl.policies      import random_policy, greedy_capture_policy, defensive_policy
from rl.game_analysis import (
    count_exposure, count_primed,
    _has_big_cluster, _simulate_enemy_fire, _move_risk_score,
)


# ======================================================================
# REWARD / PENALTY WEIGHTS
# Terminal reward +5.0 is set in env.py.
# ======================================================================

# -- Per-move rewards --
REWARD_CLEAN_CAPTURE    =  0.035   # per enemy orb captured without increasing exposure
REWARD_CASCADE_OWN      =  0.012   # per net orb gained beyond the 1 placed (cascade payoff)

# -- Positional rewards --
# REWARD_CHAIN and REWARD_ATTACK_CHAIN use "full current score when it grows" logic:
# growing a longer chain rewards more than starting a short one.
# REWARD_ATTACK_CHAIN is intentionally low — it stacks on top of REWARD_CHAIN for the
# same pairs, so the combined signal for attacking chain growth is CHAIN + ATTACK_CHAIN.
# All other positional rewards are true delta (net-new this turn only).
REWARD_SAFE_PRIME       =  0.050   # per net-new safe primed cell (not adj to enemy primed) — stepping stone to chains
REWARD_CHAIN            =  0.060   # full chain-pair score when chain grows
REWARD_SAFE_ATTACK      =  0.055   # per net-new primed adj to UNPRIMED enemy (not exposed)
REWARD_ATTACK_CHAIN     =  0.040   # full attacking chain score when it grows (stacks with CHAIN — kept low)
REWARD_CORNER_EXECUTE   =  0.080   # corner blast executed: corner was primed + adjacent owned pre-move, now fired
REWARD_EDGE_CHAIN_BONUS =  0.045   # per net-new chain pair where >= 1 cell is on the board edge

# -- Tactical rewards --
# Hub prime requires >= 3 primed neighbours to avoid firing on linear chains.
# A cell with 3 primed neighbours is a dense cluster junction — when it fires,
# it triggers 3 simultaneous chain reactions. Genuinely rare and powerful.
REWARD_HUB_PRIME          =  0.060  # per net-new primed with >= 3 primed neighbours

# Corner control: fires when newly owning any corner cell (cm=2, easiest to prime).
# Small value — it's an early-game stepping stone toward CORNER_SETUP, not a goal itself.
REWARD_CORNER_CONTROL     =  0.015  # per newly claimed corner cell — phase-scaled (3× early, 1× mid)

# L-overload execution reward (no build-up signal — too noisy from cascade accidents).
REWARD_L_OVERLOAD_EXECUTE =  0.15   # per enemy primed destroyed during L-overload execution

# -- Sim-cascade penalty cap --
# Prevents a single heavily-exposed position from generating a -3.0+ penalty
# that overwhelms every positive signal and teaches the model to never be near the enemy.
SIM_CASCADE_MAX_PENALTY   =  0.45   # hard cap on Type 1 sim-cascade total penalty per step

# -- Exposure resolution reward (Type 2) --
REWARD_EXPOSURE_GAIN    =  0.10    # per enemy primed destroyed when force-greedy resolves exposure

# -- Penalties --
PENALTY_WASTED_PLACEMENT    =  0.20   # placing non-priming orb next to enemy primed — dumbest move possible
PENALTY_SIM_CASCADE_LOSS    =  0.15   # per orb lost in simulated enemy fire (Type 1)
PENALTY_EXPOSURE_MOVE       =  0.25   # pushed to buffer for each action blocked by risk filter

# -- Chain reward cap --
# Limits the "full current score" payout for CHAIN and ATTACK_CHAIN rewards.
# Beyond this many pairs the per-move reward plateaus, preventing Q-value inflation
# from unbounded chain growth while preserving the "longer chains pay more" incentive.
CHAIN_REWARD_CAP         =  5      # max chain pairs counted in full-score reward

# -- Any-prime stepping-stone reward (unsafe primes included) --
# Gives a minimal signal for simply priming a cell even when exposed.
# Safe primes earn REWARD_SAFE_PRIME + REWARD_ANY_PRIME combined; unsafe earn only this.
REWARD_ANY_PRIME         =  0.001  # per net-new primed cell regardless of safety


# -- Unprimed cell penalty --
# Fires in midgame (total_orbs > 10) when player owns more than 5 unprimed cells.
# Penalises scatter-and-hold: owning cells with single orbs provides no chain/prime value.
# Scales linearly per excess cell so large unprimed sprawls are punished proportionally.
PENALTY_UNPRIMED_CELLS   =  0.030  # per owned unprimed cell above the threshold of 5


# -- Per-stage learning rate drops at graduation --
# Loss was monotonically increasing in Stage 2 with the Stage 1 LR (5e-5).
# Dropping LR at each graduation stabilises the Bellman targets on harder opponents.
STAGE_LR        = {1: 4e-5, 2: 2e-5, 3: 2e-5, 4: 1e-5}

# -- Partial epsilon reset at graduation (don't throw away Stage N knowledge) --
# Resetting to 1.0 floods the buffer with random transitions right when the
# model needs stable replay from the new, harder opponent.
STAGE_EPS_START = {2: 0.80, 3: 0.60, 4: 0.45}
# Stage 2: 0.80 — must earn its way down to GRAD_MIN_EPS[2]=0.50 over ~2000 eps.
# Stage 3: 0.60 — model arrives with basic tactics, slightly less exploration needed.
# Stage 4: 0.45 — consolidation phase; pull back from the crash-prone 0.28 that post-
#           graduation would otherwise inherit. Gives the model breathing room before
#           epsilon decays to eps_end.

# -- Per-stage target network update frequency --
# Less frequent updates = more stable TD reference = crash amplification damped.
# Stage 3 raised to 1500 and Stage 4 to 2000 to reduce Q-value divergence severity
# when epsilon decays into the exploitation zone.
STAGE_TARGET_UPDATE = {1: 1000, 2: 1000, 3: 1500, 4: 2000}

# ======================================================================
# CURRICULUM THRESHOLDS
# ======================================================================

GRAD_RANDOM     = 0.70
GRAD_DEFENSIVE  = 0.68
GRAD_GREEDY     = 0.40

CURRICULUM_MIN_EPS = 300   # minimum stage episodes before graduation check

# Per-stage minimum epsilon before graduation is even checked.
# Stage 1: eps 1.0→0.80  (~1100 eps, same as before)
# Stage 2: eps 0.80→0.60 (~2000 eps with eps_decay_2=150k, ~30 steps/ep)
# Stage 3: eps 0.60→0.45 (~3600 eps with eps_decay_3=200k, before graduating)
GRAD_MIN_EPS = {1: 0.80, 2: 0.60, 3: 0.45}


# ------------------------------------------------------------------
# Epsilon schedule
# ------------------------------------------------------------------

def get_epsilon(step, eps_start, eps_end, eps_decay):
    return eps_start + min(step / eps_decay, 1.0) * (eps_end - eps_start)


def _exec_safe_action(env, player, mask, agent, state, epsilon):
    """
    Agent picks an action; if _move_risk_score returns > 0 (our loss exceeds our
    gain from the trade), block it, push a penalty transition so those Q-values
    get a direct negative gradient, and retry.
    Falls back to the least-risky available move if all options are net negative.
    Filter is inactive before total_orbs >= 28 (handled inside _move_risk_score).
    Returns (action, r_act, c_act).
    """
    remaining   = mask.copy()
    risk_scores = {}

    while True:
        action       = agent.select_action(state, remaining, epsilon)
        r_act, c_act = divmod(action, env.game.cols)
        risk         = _move_risk_score(env.game, player, r_act, c_act)

        if risk == 0.0:
            return action, r_act, c_act   # acceptable trade

        # Simulate the risky move → capture bad next_state → push penalty transition
        snap = env.game._snapshot()
        env.game.grid[r_act][c_act].owner  = player
        env.game.grid[r_act][c_act].count += 1
        if env.game.grid[r_act][c_act].count >= env.game.critical_mass(r_act, c_act):
            env.game.state = "animating"
            env.game.explosion_queue.append((r_act, c_act))
            sfty = 0
            while env.game.state == "animating" and sfty < 10_000:
                wave = env.game.get_wave()
                if not wave:
                    break
                env.game.apply_wave(wave)
                sfty += 1
        bad_next = env.encode_state(1 - player)
        bad_msk  = env.valid_action_mask(1 - player)
        env.game._restore(snap)

        agent.push(state, action, -PENALTY_EXPOSURE_MOVE, bad_next, False, bad_msk)

        risk_scores[action] = risk
        remaining[action]   = False

        if not remaining.any():
            # All moves exceed threshold — pick the least damaging one
            best           = min(risk_scores, key=risk_scores.get)
            best_r, best_c = divmod(best, env.game.cols)
            return best, best_r, best_c


# ------------------------------------------------------------------
# Positional snapshot  (captures board state before a move for delta rewards)
# ------------------------------------------------------------------

def _positional_snapshot(game, player):
    """
    Snapshot of all positional metrics for `player` before a move.
    Used by compute_shaping to compute deltas and detect execution events.

    Fields
    ------
    chain_pairs        : primed-to-primed adjacent pairs (// 2)
    safe_attack        : count of primed cells adj to UNPRIMED enemy, not exposed
    attack_chain_score : sum of REWARD_ATTACK_CHAIN/2 over attacking chain pairs (both dirs)
    corner_setup_cells : frozenset of corner (r,c) that are primed + have >= 1 adjacent cell owned — ready to execute
    edge_chain_pairs   : chain pairs where >= 1 cell is on the board edge
    hub_primes         : primed cells with >= 3 primed neighbours (dense cluster nodes only)
    l_overload         : count of 2x2 configs with 3 my primed + 4th (any owner) at >= cm-2
                         adjacent to enemy primed — ready to execute
    enemy_primed_count : total enemy primed cells (used to detect L-overload execution)
    """
    enemy        = 1 - player
    my_primed    = set()
    enemy_primed = set()
    for r in range(game.rows):
        for c in range(game.cols):
            cell = game.grid[r][c]
            if cell.owner == player and game.is_primed(r, c):
                my_primed.add((r, c))
            elif cell.owner == enemy and game.is_primed(r, c):
                enemy_primed.add((r, c))

    enemy_cells = {(r, c) for r in range(game.rows) for c in range(game.cols)
                   if game.grid[r][c].owner == enemy}

    # Safe primed: my primed cells not adjacent to any enemy primed (not exposed).
    safe_primes = sum(
        1 for (r, c) in my_primed
        if not any(nb in enemy_primed for nb in game.neighbours(r, c))
    )

    chain_pairs = sum(
        1 for (r, c) in my_primed
        for nb in game.neighbours(r, c) if nb in my_primed
    ) // 2

    attacking = {(r, c) for (r, c) in my_primed
                 if any(nb in enemy_cells for nb in game.neighbours(r, c))}

    safe_attacking = {(r, c) for (r, c) in attacking
                      if not any(nb in enemy_primed for nb in game.neighbours(r, c))}

    attack_chain_score = sum(
        REWARD_ATTACK_CHAIN / 2
        for (r, c) in my_primed
        for nb in game.neighbours(r, c)
        if nb in my_primed and ((r, c) in attacking or nb in attacking)
    )

    corners = {(0, 0), (0, game.cols-1), (game.rows-1, 0), (game.rows-1, game.cols-1)}

    corner_orbs = sum(1 for (r, c) in corners if game.grid[r][c].owner == player)

    corner_setup_cells = frozenset(
        (r, c) for (r, c) in corners
        if (r, c) in my_primed
        and any(game.grid[nr][nc].owner == player
                for nr, nc in game.neighbours(r, c))
    )

    edge_cells = {(r, c) for r in range(game.rows) for c in range(game.cols)
                  if r == 0 or r == game.rows - 1 or c == 0 or c == game.cols - 1}
    edge_chain_pairs = sum(
        1 for (r, c) in my_primed
        for nb in game.neighbours(r, c)
        if nb in my_primed and ((r, c) in edge_cells or nb in edge_cells)
    ) // 2

    # Hub primes: >= 3 primed neighbours required to exclude linear chains.
    # In a 6x6 board, an interior cell (4 neighbours) qualifies only when 3+ are primed.
    # Edge cells (3 neighbours) need all 3 primed. Corner cells (2 neighbours) can never qualify.
    hub_primes = sum(
        1 for (r, c) in my_primed
        if sum(1 for nb in game.neighbours(r, c) if nb in my_primed) >= 3
    )

    # L-overload config detection.
    # 4th cell (any owner) must be at >= cm-2 so 2 incoming orbs from the 2x2 wave
    # push it to critical mass and trigger an explosion into adjacent enemy primed.
    # Ownership is irrelevant — the first incoming orb converts it regardless.
    l_overload = 0
    for r in range(game.rows - 1):
        for c in range(game.cols - 1):
            square       = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
            my_primed_sq = [pos for pos in square if pos in my_primed]
            if len(my_primed_sq) != 3:
                continue
            fourth = next(pos for pos in square if pos not in my_primed_sq)
            fr, fc = fourth
            cell_f = game.grid[fr][fc]
            # Must be close enough to critical mass to fire when hit by 2 orbs
            if cell_f.count < game.critical_mass(fr, fc) - 2:
                continue
            # Must be adjacent to at least one enemy primed cell (that's what we're blasting)
            if any(nb in enemy_primed for nb in game.neighbours(fr, fc)):
                l_overload += 1

    return {
        'all_primes':         len(my_primed),
        'safe_primes':        safe_primes,
        'chain_pairs':        chain_pairs,
        'safe_attack':        len(safe_attacking),
        'attack_chain_score': attack_chain_score,
        'corner_orbs':        corner_orbs,
        'corner_setup_cells': corner_setup_cells,
        'edge_chain_pairs':   edge_chain_pairs,
        'hub_primes':         hub_primes,
        'l_overload':         l_overload,
        'enemy_primed_count': len(enemy_primed),
    }


# ------------------------------------------------------------------
# Shaping reward  (Type 1 exposure penalty lives here)
# ------------------------------------------------------------------

def compute_shaping(game, player, counts_before, counts_after, pos_before,
                    exposure_before=0):
    """
    Full per-step shaping reward, computed after the move has resolved.
    Includes all positional rewards and the Type 1 sim-cascade penalty.
    Does NOT include wasted-placement or Type 2 exposure (those live in runners).

    exposure_before : count_exposure(game, player) BEFORE the move.

    REWARD_CLEAN_CAPTURE only fires when the move does not increase exposure
    (len(exposed) <= exposure_before). Dirty captures earn nothing — the exposure
    penalties already handle those.

    L-overload execution fires when:
      - An L-overload config existed pre-move (pos_before['l_overload'] > 0)
      - Enemy primed cells were destroyed post-cascade
    Scaled per enemy primed destroyed so deeper executions earn proportionally more.
    """
    enemy  = 1 - player
    reward = 0.0

    # -- Build current primed sets --
    my_primed    = set()
    enemy_primed = set()
    for r in range(game.rows):
        for c in range(game.cols):
            cell = game.grid[r][c]
            if cell.owner == player and game.is_primed(r, c):
                my_primed.add((r, c))
            elif cell.owner == enemy and game.is_primed(r, c):
                enemy_primed.add((r, c))

    enemy_cells = {(r, c) for r in range(game.rows) for c in range(game.cols)
                   if game.grid[r][c].owner == enemy}

    exposed = {(r, c) for (r, c) in my_primed
               if any(nb in enemy_primed for nb in game.neighbours(r, c))}
    clean   = len(exposed) <= exposure_before

    # -- Per-move terms --
    enemy_captured = max(counts_before[enemy] - counts_after[enemy], 0)
    if clean and enemy_captured:
        reward += REWARD_CLEAN_CAPTURE * enemy_captured

    # cascade_gain: net orbs gained beyond the 1 orb placed this turn
    cascade_gain = max((counts_after[player] - counts_before[player]) - 1, 0)
    reward += REWARD_CASCADE_OWN * cascade_gain

    # -- Phase boost: early-game rewards worth more when the board is sparse --
    # Decays from 3.0× at game start to 1.0× at ~12 total orbs on board.
    # Incentivises corner/priming strategy in the first few moves of each game.
    total_orbs   = sum(counts_after)
    phase_boost  = max(1.0, 3.0 - total_orbs / 10.0)

    # -- Safe primed cells (delta): stepping stone toward chain discovery --
    safe_primes = sum(
        1 for (r, c) in my_primed
        if not any(nb in enemy_primed for nb in game.neighbours(r, c))
    )
    reward += REWARD_SAFE_PRIME * max(safe_primes - pos_before['safe_primes'], 0) * phase_boost

    # -- Any new primed cell (delta): signal even for unsafe primes --
    all_primes = len(my_primed)
    reward += REWARD_ANY_PRIME * max(all_primes - pos_before.get('all_primes', 0), 0) * phase_boost

    # -- Chain pairs: full current score if chain grew --
    chain_pairs = sum(
        1 for (r, c) in my_primed
        for nb in game.neighbours(r, c) if nb in my_primed
    ) // 2
    if chain_pairs > pos_before['chain_pairs']:
        reward += REWARD_CHAIN * min(chain_pairs, CHAIN_REWARD_CAP)

    # -- Safe attack (delta): primed adj to unprimed enemy, not exposed --
    attacking = {(r, c) for (r, c) in my_primed
                 if any(nb in enemy_cells for nb in game.neighbours(r, c))}
    safe_attacking = {(r, c) for (r, c) in attacking
                      if not any(nb in enemy_primed for nb in game.neighbours(r, c))}
    reward += REWARD_SAFE_ATTACK * max(len(safe_attacking) - pos_before['safe_attack'], 0)

    # -- Attack chain: full current score if attacking chain grew --
    attack_chain_score = sum(
        REWARD_ATTACK_CHAIN / 2
        for (r, c) in my_primed
        for nb in game.neighbours(r, c)
        if nb in my_primed and ((r, c) in attacking or nb in attacking)
    )
    if attack_chain_score > pos_before['attack_chain_score']:
        reward += min(attack_chain_score, REWARD_ATTACK_CHAIN * CHAIN_REWARD_CAP)

    # -- Corner control (delta): any newly claimed corner cell --
    corners = {(0, 0), (0, game.cols-1), (game.rows-1, 0), (game.rows-1, game.cols-1)}
    corner_orbs = sum(1 for (r, c) in corners if game.grid[r][c].owner == player)
    reward += REWARD_CORNER_CONTROL * max(corner_orbs - pos_before['corner_orbs'], 0) * phase_boost

    # -- Corner execute: setup existed pre-move and corner has now fired --
    # A corner fires when it was primed (count==1, cm==2) and received an orb;
    # post-cascade its count is no longer 1.
    corners_executed = sum(
        1 for (r, c) in pos_before.get('corner_setup_cells', frozenset())
        if game.grid[r][c].count != game.critical_mass(r, c) - 1
    )
    reward += REWARD_CORNER_EXECUTE * corners_executed * phase_boost

    # -- Edge chain bonus (delta): chain pairs where >= 1 cell is on board edge --
    edge_cells = {(r, c) for r in range(game.rows) for c in range(game.cols)
                  if r == 0 or r == game.rows - 1 or c == 0 or c == game.cols - 1}
    edge_chain_pairs = sum(
        1 for (r, c) in my_primed
        for nb in game.neighbours(r, c)
        if nb in my_primed and ((r, c) in edge_cells or nb in edge_cells)
    ) // 2
    reward += REWARD_EDGE_CHAIN_BONUS * max(edge_chain_pairs - pos_before['edge_chain_pairs'], 0)

    # -- Hub prime (delta): primed cells with >= 3 primed neighbours --
    hub_primes = sum(
        1 for (r, c) in my_primed
        if sum(1 for nb in game.neighbours(r, c) if nb in my_primed) >= 3
    )
    reward += REWARD_HUB_PRIME * max(hub_primes - pos_before['hub_primes'], 0)

    # -- L-overload EXECUTION reward --
    # Config existed pre-move AND enemy primed were destroyed post-cascade.
    # Fires per enemy primed destroyed: a 3-primed-destroyed execution earns 3x a 1-primed one.
    if pos_before['l_overload'] > 0:
        enemy_primed_destroyed = pos_before['enemy_primed_count'] - len(enemy_primed)
        if enemy_primed_destroyed > 0:
            reward += REWARD_L_OVERLOAD_EXECUTE * enemy_primed_destroyed

    # -- TYPE 1 EXPOSURE PENALTY --
    # For each enemy primed cell threatening an exposed primed of ours,
    # simulate it firing and measure our orb loss. Penalise per orb lost.
    if exposed:
        threatening    = {nb for (r, c) in exposed
                          for nb in game.neighbours(r, c) if nb in enemy_primed}
        total_sim_loss = sum(_simulate_enemy_fire(game, er, ec, player)
                             for (er, ec) in threatening)
        reward -= min(PENALTY_SIM_CASCADE_LOSS * total_sim_loss, SIM_CASCADE_MAX_PENALTY)

    # -- UNPRIMED CELL PENALTY (midgame) --
    # Only fires after the early game so normal territory claiming isn't punished.
    # Owned cells that aren't primed contribute nothing to chains or attacks.
    if total_orbs > 10:
        owned_cells    = sum(1 for r in range(game.rows) for c in range(game.cols)
                             if game.grid[r][c].owner == player)
        owned_unprimed = owned_cells - len(my_primed)
        if owned_unprimed > 5:
            reward -= PENALTY_UNPRIMED_CELLS * (owned_unprimed - 5)

    return reward


# ------------------------------------------------------------------
# Wasted placement penalty  (fires pre-step in runners)
# ------------------------------------------------------------------

def wasted_placement_penalty(game, player, r, c):
    """
    Returns -PENALTY_WASTED_PLACEMENT if the placement at (r, c) drops a
    non-priming orb next to an enemy primed cell (a sitting duck).
    Returns 0.0 if the placement primes or explodes the cell (intentional play).
    """
    pre_cell  = game.grid[r][c]
    cm        = game.critical_mass(r, c)
    new_count = pre_cell.count + 1
    if new_count >= cm or new_count == cm - 1:
        return 0.0
    enemy        = 1 - player
    enemy_primed = {(rr, cc) for rr in range(game.rows) for cc in range(game.cols)
                    if game.grid[rr][cc].owner == enemy and game.is_primed(rr, cc)}
    if any(nb in enemy_primed for nb in game.neighbours(r, c)):
        return -PENALTY_WASTED_PLACEMENT
    return 0.0


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def _eval_vs_policy(agent, env, policy_fn, n_games=200):
    wins = 0
    for i in range(n_games):
        agent_seat = i % 2
        state, _   = env.reset()
        done       = False
        guard      = env.rows * env.cols * 10
        while not done and guard > 0:
            p    = env.game.current_player
            mask = env.valid_action_mask(p)
            if p == agent_seat:
                action = agent.select_action(state, mask, epsilon=0.0)
            else:
                r, c   = policy_fn(env.game, p)
                action = r * env.cols + c
            state, _, done, _, _ = env.step(action)
            guard -= 1
        if env.game.winner == agent_seat:
            wins += 1
    return wins / n_games


def evaluate_vs_random(agent, env, n=200):
    return _eval_vs_policy(agent, env, random_policy, n)

def evaluate_vs_greedy(agent, env, n=200):
    return _eval_vs_policy(agent, env, greedy_capture_policy, n)

def evaluate_vs_defensive(agent, env, n=200):
    return _eval_vs_policy(agent, env, defensive_policy, n)


# ------------------------------------------------------------------
# Corner opening hardcode helpers
# ------------------------------------------------------------------

def _get_clean_corner(game, player):
    """
    Return the first corner (TL → TR → BL → BR) whose cell and all its
    neighbours are enemy-free.  Returns None if no such corner exists.
    Used on the agent's first turn; the board is empty so this always
    succeeds, but the check makes it robust for any future call site.
    """
    enemy   = 1 - player
    corners = [
        (0,            0),
        (0,            game.cols - 1),
        (game.rows - 1, 0),
        (game.rows - 1, game.cols - 1),
    ]
    for cr, cc in corners:
        zone = [(cr, cc)] + list(game.neighbours(cr, cc))
        if all(game.grid[r][c].owner != enemy for r, c in zone):
            return (cr, cc)
    return None


def _corner_hardcode_action(game, player, agent_turn, chosen_corner, corner_adj):
    """
    Deterministic opening for the agent's first 3 turns:
      Turn 0 — place on a clean corner            → corner becomes primed (count=1, cm=2)
      Turn 1 — place on an adjacent edge cell     → sets up the blast
      Turn 2 — place on the corner again          → corner explodes, cascades to adj

    Returns (action_int, new_chosen_corner, new_corner_adj) on success,
    or None if the plan can't proceed (corner captured, adj taken, etc.).
    All callers must still verify the returned action is in the valid mask.
    """
    cols  = game.cols
    enemy = 1 - player

    if agent_turn == 0:
        corner = _get_clean_corner(game, player)
        if corner is None:
            return None
        cr, cc = corner
        return cr * cols + cc, corner, None

    if chosen_corner is None:
        return None
    cr, cc = chosen_corner

    if agent_turn == 1:
        # Corner must still be ours (or empty — we placed there last turn)
        if game.grid[cr][cc].owner == enemy:
            return None
        # Adjacent cells: prefer edge cells (not corners), then any non-enemy cell
        neighbours = list(game.neighbours(cr, cc))
        candidates = [(r, c) for r, c in neighbours if game.grid[r][c].owner != enemy]
        if not candidates:
            return None
        edge_cands = [
            (r, c) for r, c in candidates
            if r == 0 or r == game.rows - 1 or c == 0 or c == game.cols - 1
        ]
        adj = edge_cands[0] if edge_cands else candidates[0]
        return adj[0] * cols + adj[1], chosen_corner, adj

    if agent_turn == 2:
        # Corner must be ours and primed (count == 1, cm == 2)
        cell = game.grid[cr][cc]
        if cell.owner != player or cell.count != game.critical_mass(cr, cc) - 1:
            return None
        return cr * cols + cc, chosen_corner, corner_adj

    return None


# ------------------------------------------------------------------
# Episode runner — stages 1-3 (vs fixed policy)
# ------------------------------------------------------------------

def _run_episode_vs_fixed(agent, env, epsilon, policy_fn, agent_seat, stage):
    """
    1-step zero-sum Bellman.  Opponent plays policy_fn.

    Type 2 exposure handling
    ------------------------
    If exposure existed at the START of the agent's turn AND a big mixed primed
    cluster (>= 4 cells) is present:
      - Action is overridden to greedy_capture_policy (force fire).
      - If exposure reduces: +REWARD_EXPOSURE_GAIN * enemy_primed_destroyed.
    """
    state, _ = env.reset()
    done     = False
    moves    = 0
    losses   = []
    guard    = env.rows * env.cols * 10
    agent_turn_count = 0   # counts agent's own turns (not total plies) for corner hardcode
    chosen_corner    = None
    corner_adj_cell  = None

    while not done and moves < guard:
        player = env.game.current_player
        mask   = env.valid_action_mask(player)

        if player != agent_seat:
            r, c   = policy_fn(env.game, player)
            action = r * env.cols + c
            state, _, done, _, _ = env.step(action)
            moves += 1
            continue

        # ---- Agent's turn ----
        exposure_before  = count_exposure(env.game, player)
        enemy            = 1 - player
        enemy_primed_pre = count_primed(env.game, enemy)
        pos_before       = _positional_snapshot(env.game, player)
        counts_before    = env.game.orb_counts()

        force_greedy = (
            exposure_before > 0
            and _has_big_cluster(env.game, player)
        )

        if force_greedy:
            # Emergency: fire into the cluster — overrides everything
            r_act, c_act = greedy_capture_policy(env.game, player)
            action       = r_act * env.cols + c_act
        elif agent_turn_count < 3 and stage <= 2:
            # Corner opening hardcode for first 3 agent turns
            hc = _corner_hardcode_action(
                env.game, player, agent_turn_count, chosen_corner, corner_adj_cell)
            if hc is not None:
                hc_action, chosen_corner, corner_adj_cell = hc
                if mask[hc_action]:
                    action       = hc_action
                    r_act, c_act = divmod(action, env.game.cols)
                else:
                    action, r_act, c_act = _exec_safe_action(
                        env, player, mask, agent, state, epsilon)
            else:
                action, r_act, c_act = _exec_safe_action(
                    env, player, mask, agent, state, epsilon)
        else:
            action, r_act, c_act = _exec_safe_action(
                env, player, mask, agent, state, epsilon)

        agent_turn_count += 1

        wasted = wasted_placement_penalty(env.game, player, r_act, c_act)

        next_state, reward, done, _, _ = env.step(action)
        moves += 1

        if done:
            r_t = reward + wasted
            agent.push(state, action, r_t, next_state, done, mask)
        else:
            exposure_after    = count_exposure(env.game, player)
            enemy_primed_post = count_primed(env.game, enemy)
            primed_destroyed  = max(enemy_primed_pre - enemy_primed_post, 0)

            r_t  = reward
            r_t += compute_shaping(env.game, player,
                                   counts_before, env.game.orb_counts(),
                                   pos_before, exposure_before)
            r_t += wasted

            if exposure_before > 0 and exposure_after < exposure_before:
                # Resolved — bonus if we forced a greedy fire
                if force_greedy:
                    r_t += REWARD_EXPOSURE_GAIN * primed_destroyed

            agent.push(state, action, r_t, next_state, done, mask)

        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

        state = next_state

    return {"winner": env.game.winner, "length": moves, "losses": losses}


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(args):
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    env = ChainReactionEnv(rows=args.rows, cols=args.cols, num_players=2)
    agent = DQNAgent(
        rows          = args.rows,
        cols          = args.cols,
        channels      = args.channels,
        lr            = args.lr,
        gamma         = args.gamma,
        buffer_size   = args.buffer_size,
        batch_size    = args.batch_size,
        target_update = args.target_update,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load(args.resume)

    stage             = 1
    stage_ep          = 0
    consec_above      = 0
    last_vs_opp       = 0.0
    stage_names       = {1: "rand", 2: "def", 3: "greedy", 4: "greedy"}
    stage_thresholds  = {1: GRAD_RANDOM, 2: GRAD_DEFENSIVE, 3: GRAD_GREEDY}
    stage_eval_fns    = {1: evaluate_vs_random, 2: evaluate_vs_defensive, 3: evaluate_vs_greedy}
    stage_policies    = {1: random_policy, 2: defensive_policy, 3: greedy_capture_policy}
    stage_eps_decays  = {1: args.eps_decay_1, 2: args.eps_decay_2, 3: args.eps_decay_3}

    current_eps_start = args.eps_start   # updated at each graduation (partial reset)
    loss_history      = deque(maxlen=500)
    start_time        = time.time()
    total_steps       = agent.steps_done
    stage_start_steps = agent.steps_done
    # Sliding-window buffer: tracks how many transitions belonged to the previous
    # stage so we can evict them at the next graduation.
    # S1→S2: record N1, no clear  (S2 trains on S1+S2)
    # S2→S3: evict first N1 entries, keep S2    (S3 trains on S2+S3)
    # S3→S4: evict first N2 entries, keep S3    (S4 trains on S3+S4)
    prev_stage_buf_len = 0   # number of transitions from the stage before current

    print(f"\nTraining  {args.rows}x{args.cols}  |  {args.episodes} episodes")
    print(f"channels={args.channels}  device={agent.device}  lr={args.lr}  gamma={args.gamma}")
    print(f"eps_decay  s1={args.eps_decay_1:,}  s2={args.eps_decay_2:,}  s3={args.eps_decay_3:,}  "
          f"target_update  s1-2={STAGE_TARGET_UPDATE[1]}  s3={STAGE_TARGET_UPDATE[3]}  s4={STAGE_TARGET_UPDATE[4]}")
    print(f"Grad thresholds — rand:{GRAD_RANDOM:.0%}  def:{GRAD_DEFENSIVE:.0%}  greedy:{GRAD_GREEDY:.0%}\n")
    print(f"{'Ep':>7}  {'Stg':>6}  {'vs_opp':>7}  {'avg_loss':>9}  "
          f"{'eps':>6}  {'steps':>8}  {'buf':>7}  {'t':>5}")
    print("-" * 72)

    for ep in range(1, args.episodes + 1):
        stage_steps = agent.steps_done - stage_start_steps
        eps         = get_epsilon(stage_steps, current_eps_start, args.eps_end,
                                  stage_eps_decays[min(stage, 3)])
        stage_ep   += 1

        run_stage = min(stage, 3)
        if stage > 3:
            # Stage 4: pure greedy — deepens and stabilises Stage 3 greedy-fighting skills.
            # Avoids gradient interference from the defensive/greedy mix.
            opp_policy = greedy_capture_policy
        else:
            opp_policy = stage_policies[run_stage]
        result    = _run_episode_vs_fixed(
            agent, env, eps, opp_policy, ep % 2, run_stage)

        total_steps = agent.steps_done
        loss_history.extend(result["losses"])

        if ep % args.eval_every == 0:
            eval_stage  = min(stage, 3)
            last_vs_opp = stage_eval_fns[eval_stage](agent, env, n=200)
            if stage <= 3 and stage_ep >= CURRICULUM_MIN_EPS and eps <= GRAD_MIN_EPS[stage]:
                if last_vs_opp >= stage_thresholds[stage]:
                    consec_above += 1
                else:
                    consec_above = 0
                if consec_above >= 2:
                    print(f"\n  *** Stage {stage}→{stage+1}: graduated ep {ep} "
                          f"(win={last_vs_opp:.1%}, eps={eps:.3f}) ***\n")
                    stage             += 1
                    stage_ep           = 0
                    consec_above       = 0
                    last_vs_opp        = 0.0
                    stage_start_steps  = agent.steps_done
                    # Sliding-window buffer eviction:
                    #   S1→S2 (stage==2): record Stage 1 size, no eviction yet.
                    #   S2→S3 (stage==3): evict Stage 1 entries, keep Stage 2.
                    #   S3→S4 (stage==4): evict Stage 2 entries, keep Stage 3.
                    #   S4+  : no eviction — same greedy opponent, all data transfers.
                    if stage == 2:
                        prev_stage_buf_len = len(agent.buffer)
                        print(f"  -> Buffer: keeping {prev_stage_buf_len:,} Stage 1 transitions for Stage 2")
                    elif stage in (3, 4):
                        cur_len    = len(agent.buffer)
                        n_to_keep  = cur_len - prev_stage_buf_len
                        if n_to_keep > 0 and prev_stage_buf_len > 0:
                            keep = list(agent.buffer.buffer)[-n_to_keep:]
                            agent.buffer.buffer.clear()
                            agent.buffer.buffer.extend(keep)
                            print(f"  -> Buffer: evicted {prev_stage_buf_len:,} old transitions, "
                                  f"kept {n_to_keep:,} Stage {stage-1} transitions")
                        prev_stage_buf_len = n_to_keep
                    # Drop LR for the new stage (stage already incremented above)
                    new_lr = STAGE_LR.get(stage, STAGE_LR[4])
                    for pg in agent.optimizer.param_groups:
                        pg['lr'] = new_lr
                    # Update target network frequency for the new stage
                    agent.target_update = STAGE_TARGET_UPDATE.get(stage, STAGE_TARGET_UPDATE[4])
                    # Partial epsilon reset: use preset for Stages 2-3; for Stage 4+
                    # keep current epsilon so we never reset back to random play.
                    current_eps_start = STAGE_EPS_START.get(stage, eps)
                    print(f"  -> LR → {new_lr:.1e}  eps_start → {current_eps_start:.2f}  "
                          f"target_update → {agent.target_update}")

        if ep % args.log_every == 0:
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0
            elapsed  = time.time() - start_time
            print(f"{ep:7d}  {stage_names[min(stage,4)]:>6}  {last_vs_opp*100:6.1f}%  "
                  f"{avg_loss:9.5f}  {eps:6.3f}  {total_steps:8d}  "
                  f"{len(agent.buffer):7d}  {elapsed:5.0f}s")

        if ep % args.save_every == 0:
            path = os.path.join(ckpt_dir, f"ep{ep:05d}.pt")
            agent.save(path)
            print(f"  -> Saved {path}")

    final_path = os.path.join(ckpt_dir, "final.pt")
    agent.save(final_path)

    print(f"\nDone.  Final checkpoint: {final_path}")
    print(f"Stage reached: {stage_names[min(stage, 4)]}")
    print("\nFinal evaluation (200 games each, alternating seats):")
    print("-" * 36)
    wr_rand = evaluate_vs_random(agent,     env, n=200)
    wr_def  = evaluate_vs_defensive(agent,  env, n=200)
    wr_grdy = evaluate_vs_greedy(agent,     env, n=200)
    print(f"  vs random    : {wr_rand:.1%}")
    print(f"  vs defensive : {wr_def:.1%}")
    print(f"  vs greedy    : {wr_grdy:.1%}")
    print("-" * 36)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3-stage curriculum DQN for Chain Reaction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rows",          type=int,   default=6)
    p.add_argument("--cols",          type=int,   default=6)
    p.add_argument("--channels",      type=int,   default=64)
    p.add_argument("--episodes",      type=int,   default=10000)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--gamma",         type=float, default=0.97)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--buffer_size",   type=int,   default=500_000)
    p.add_argument("--target_update", type=int,   default=1000)
    p.add_argument("--eps_start",     type=float, default=1.0)
    p.add_argument("--eps_end",       type=float, default=0.05)
    p.add_argument("--eps_decay_1",   type=int,   default=150_000)
    p.add_argument("--eps_decay_2",   type=int,   default=150_000)
    p.add_argument("--eps_decay_3",   type=int,   default=200_000)
    p.add_argument("--log_every",     type=int,   default=50)
    p.add_argument("--eval_every",    type=int,   default=50)
    p.add_argument("--save_every",    type=int,   default=500)
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint to resume training from")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
