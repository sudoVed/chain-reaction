# Chain Reaction — Reinforcement Learning Technical Reference

This document covers every aspect of the RL training system: architecture, state
encoding, training loop, reward shaping, policies, hardcoded behavioral overrides,
evaluation tools, and checkpoint history. It is intended as the authoritative
reference for anyone modifying or re-running the training pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [File Map](#file-map)
3. [Neural Network — model.py](#neural-network--modelpy)
4. [Environment — env.py](#environment--envpy)
5. [Agent — agent.py](#agent--agentpy)
6. [Policies — policies.py](#policies--policiespy)
7. [Board Analysis — game_analysis.py](#board-analysis--game_analysispy)
8. [Training Loop — train.py](#training-loop--trainpy)
   - [Curriculum](#curriculum)
   - [Reward Shaping — All Values](#reward-shaping--all-values)
   - [Positional Snapshot](#positional-snapshot)
   - [Hardcoded Overrides](#hardcoded-overrides)
   - [Risk Filter](#risk-filter)
   - [Sliding-Window Replay Buffer](#sliding-window-replay-buffer)
   - [Hyperparameters](#hyperparameters)
9. [Debug Visualiser — debug_watch.py](#debug-visualiser--debug_watchpy)
10. [Arena Evaluator — arena.py](#arena-evaluator--arenapy)
11. [Passive Viewer — watch.py](#passive-viewer--watchpy)
12. [Checkpoints](#checkpoints)
13. [Training Run History](#training-run-history)
14. [Known Issues and Design Decisions](#known-issues-and-design-decisions)

---

## Overview

The RL system trains a DQN agent to play Chain Reaction on a 6×6 board using a
4-stage curriculum of progressively harder fixed-policy opponents. The model is a
fully-convolutional spatial CNN that outputs one Q-value per board cell. A single
network plays both sides of the board by always encoding state from the perspective
of whoever is about to move.

Training uses a zero-sum 1-step Bellman target: the next state is always from the
opponent's perspective, so their best Q-value is negated when computing the target.
Dense reward shaping provides per-step feedback throughout each game; the terminal
signal alone (+5 win / 0 other) is too sparse to drive meaningful learning.

**Final model performance (ep10000 / final.pt):**

| Opponent  | Win rate |
|-----------|----------|
| Random    | 95.0%    |
| Defensive | 81.5%    |
| Greedy    | 57.0%    |

All evaluations use 200 games with alternating seats (100 as P0, 100 as P1).

---

## File Map

```
rl/
├── model.py          Spatial CNN Q-network (DQN class)
├── env.py            Gym-style wrapper: reset, step, encode_state, valid_action_mask
├── agent.py          DQNAgent: replay buffer, epsilon-greedy, train_step, save/load
├── policies.py       Three fixed opponent policies used as curriculum stages
├── game_analysis.py  Pure board analysis: exposure, clusters, cascade sim, risk filter
├── train.py          Full training loop: reward shaping, curriculum, eval, checkpoints
├── debug_watch.py    Step-by-step reward breakdown visualiser (pygame)
├── arena.py          Headless model-vs-model evaluation with stats
├── watch.py          Passive pygame viewer: watch any agent or policy play
├── final/
│   ├── final.pt      Best checkpoint — ep10000, used in production
│   ├── ep00500.pt … ep10000.pt   Milestone checkpoints every 500 episodes
│   └── logs.txt      Complete training log for the final run
└── checkpoint5500/
    ├── final.pt      Older run — stronger vs defensive, weaker vs greedy
    ├── ep00500.pt … ep10000.pt   Milestone checkpoints from earlier run
    └── logs.txt      Training log (includes the Stage 2 crash event)
```

---

## Neural Network — model.py

**Class:** `DQN(rows, cols, channels=64)`

A fully-convolutional Q-network. All layers use same-padding (kernel 3×3, pad 1) or
1×1 convolutions, so the network is completely grid-size agnostic — weights trained
on a 6×6 board can be applied to any other size without modification.

```
Input:   (batch, 4, H, W)    — four-channel board representation
Block 1: Conv2d(4 → 32,  3×3, pad=1) + BatchNorm2d(32)  + ReLU
Block 2: Conv2d(32 → 64, 3×3, pad=1) + BatchNorm2d(64)  + ReLU
Block 3: Conv2d(64 → 64, 3×3, pad=1) + BatchNorm2d(64)  + ReLU
Head:    Conv2d(64 → 1,  1×1)
Output:  flatten → (batch, H*W)   — one Q-value per board cell
```

The `forward` method accepts either a flat `(batch, 4*H*W)` tensor (from the replay
buffer) or a proper `(batch, 4, H, W)` image tensor — it reshapes automatically.

**Why fully convolutional?** Convolutional filters share weights across all positions,
so the same pattern (e.g. "I have a primed cell adjacent to an enemy primed cell") is
recognised anywhere on the board without being position-biased. This is a strong
inductive bias for a spatial strategy game.

**Why BatchNorm?** Stabilises training across the wide range of board states seen
during curriculum. Without it, Q-value scale varies too much between early game
(sparse board) and late game (dense primed clusters), making gradient magnitudes
inconsistent.

---

## Environment — env.py

**Class:** `ChainReactionEnv(rows, cols, num_players)`

A minimal Gym-style wrapper around `game.Game`. It handles state encoding, action
masking, and cascade simulation. The environment itself is stateless in the Gym sense:
no history, no frame stacking.

### State Encoding

`encode_state(player)` returns a flat `float32` array of length `4 * rows * cols`.
The board is always encoded from the perspective of `player`, so the same network
can play either seat.

| Channel | Content | Range |
|---------|---------|-------|
| 0 | My orb counts / 4.0 | [0, 1.0] |
| 1 | Enemy orb counts / 4.0 | [0, 1.0] |
| 2 | Critical mass / 4.0 (static geometry) | {0.5, 0.75, 1.0} |
| 3 | Primed map: +1.0 my primed, −1.0 enemy primed, 0.0 else | {−1, 0, +1} |

**Channel 2 values by cell position:**
- Corner cell (2 neighbours): `2/4 = 0.5`
- Edge cell (3 neighbours): `3/4 = 0.75`
- Interior cell (4 neighbours): `4/4 = 1.0`

**Channel 3 semantics** — the CNN can directly read spatial danger from adjacencies:
- `+1` next to `−1`: my primed cell is adjacent to enemy primed (exposure / opportunity)
- `+1` next to `+1`: my chain growing
- `−1` next to `−1`: enemy chain forming nearby

The critical-mass channel is precomputed once at init time and cached in
`_cm_channel` — it never changes during a game.

### Action Space

Integer in `[0, rows*cols)` where `action = r * cols + c`. Invalid actions (cells
owned by the opponent) are masked to `−1e9` in Q-values before `argmax`, so they
are never selected.

### Terminal Reward

- `+5.0` if the player who just moved wins (returned as `reward` in `step`)
- `0.0` for all non-terminal steps
- `−1.0` for an invalid action (should never occur with masking)

All per-step shaping rewards are computed externally in `train.py`, not inside the
environment.

### Cascade Simulation

`_simulate_cascade()` drives the game's explosion queue to completion after each
placement. It has a `safety` counter (10,000 iterations) to detect infinite loops.

---

## Agent — agent.py

**Classes:** `ReplayBuffer`, `DQNAgent`

### ReplayBuffer

A fixed-capacity circular `deque`. Each entry is a 6-tuple:

```
(state, action, reward, next_state, done, next_mask)
```

`next_state` is always encoded from the **opponent's** perspective (zero-sum
convention). `next_mask` is the valid action mask for whoever moves next.

- `push(...)`: append one transition; oldest is automatically discarded when full
- `sample(batch_size)`: random mini-batch, returns six numpy arrays

### DQNAgent

Maintains two DQN networks: `q_net` (online, trained every step) and `target_net`
(frozen, used for Bellman targets, hard-copied every `target_update` steps).

**Action selection** (`select_action`): epsilon-greedy with masking. Invalid actions
are set to `−1e9` before `argmax`. Under epsilon, picks uniformly from valid actions
only.

**Bellman target (1-step zero-sum):**

```
target = r + gamma * (1 − done) * (−max Q̂(s'))
```

The negation is essential: `s'` is encoded from the opponent's perspective. A
position where they have a high Q-value is bad for us, so we negate before using it
as our target.

**Loss:** Smooth L1 (Huber loss) between current Q and target Q, averaged over the
mini-batch. Gradient clipping at `max_norm=1.0` prevents exploding gradients.

**Checkpoint format** (`.pt` file):

```python
{
    "rows":       int,
    "cols":       int,
    "q_net":      state_dict,
    "target_net": state_dict,
    "optimizer":  state_dict,
    "steps_done": int,
}
```

`from_checkpoint(path)` is a classmethod that reads `rows`/`cols` from the checkpoint
and reconstructs the correctly-sized network — useful when loading a 6×6 checkpoint
in a context that assumes a different default size.

---

## Policies — policies.py

Three stateless, deterministic fixed policies used as training opponents and as
reference opponents for evaluation. All share the signature:

```python
policy(game: Game, player: int) -> (row: int, col: int)
```

### random_policy (Stage 1 opponent)

Picks any legal cell uniformly at random. Used in Stage 1 of the curriculum.
Provides simple, dense feedback — the agent just needs to beat chaos. No strategic
content whatsoever.

**Why start here?** Random exploration by the policy fills the replay buffer with
diverse board positions during the period when epsilon is high. The agent sees a
wide variety of cascades without being punished for mistakes it doesn't yet
understand.

### greedy_capture_policy (Stage 3 / 4 opponent)

Priority-ordered selection among legal moves:

1. **My primed cell adjacent to ≥1 enemy primed** — triggers their chain too (attack and steal)
2. **Any of my primed cells** — purely builds own chains
3. **Any legal move** — fallback random

A "primed" cell is one at `count == critical_mass − 1`, meaning one more orb will
cause it to explode. The greedy policy fires aggressively, preferring chain reactions
over territorial expansion.

**Why is this harder than defensive?** It actively pushes cascades into your
territory rather than avoiding yours. The agent must learn to not leave primed cells
adjacent to the enemy, or to fire before the enemy can.

### defensive_policy (Stage 2 opponent)

Priority-ordered selection, biased toward safety:

1. **Safe own primed adjacent to enemy primed** — preemptive blast (safe position that can attack)
2. **Any safe own primed cell** — safe chain-building
3. **Any safe move** — neutral but avoids handing enemy a free cascade
4. **Dangerous own primed** (last resort) — at least attack even if risky
5. **Any legal move** — absolute fallback

A move is "dangerous" if placing on that cell would be adjacent to an enemy primed
cell (giving them a free cascade target next turn).

**Why is defensive between random and greedy?** It is harder than random because it
avoids gifts; easier than greedy because it doesn't actively trigger chains. The
agent must learn to recognise and exploit the exposure the defensive policy leaves
around its primed cells.

---

## Board Analysis — game_analysis.py

Pure analysis utilities, imported by `train.py`, `arena.py`, and `debug_watch.py`.
All functions that mutate game state use `game._snapshot()` / `game._restore()` to
avoid touching live state. Never call these without restoring.

### count_exposure(game, player) → int

Count of player's primed cells that are adjacent to at least one enemy primed cell.
A primed cell adjacent to enemy primed is "exposed" — the enemy can cascade into it
on their next turn for free.

### count_primed(game, player) → int

Simple count of all primed cells owned by player.

### _primed_components(game) → list[set]

Connected components of ALL primed cells across both players (4-connectivity BFS).
A "component" is a set of adjacent primed cells regardless of owner. Used by
`_has_big_cluster`.

### _has_big_cluster(game, player) → bool

Returns `True` if any mixed primed component has ≥4 cells AND contains ≥1 cell
owned by `player`. This is the trigger condition for the force-greedy override (see
[Hardcoded Overrides](#hardcoded-overrides)).

**Intuition:** when 4+ primed cells (from both players combined) form a connected
cluster and you own at least one of them, a chain reaction is imminent. The model
should fire immediately rather than waiting.

### _simulate_enemy_fire(game, er, ec, agent_player) → int

Snapshot, push the enemy primed cell at `(er, ec)` over critical mass, run the full
cascade headlessly, measure the agent's orb loss, restore. Returns `max(orbs_before
− orbs_after, 0)`. Used by `compute_shaping` to compute the Type 1 exposure penalty.

### _move_risk_score(game, player, r, c) → float

The risk filter — see [Risk Filter](#risk-filter) below.

---

## Training Loop — train.py

The main orchestration file. Runs the curriculum, computes shaping rewards, manages
buffer eviction, and saves checkpoints.

### Curriculum

Training progresses through 4 stages. Each stage uses a different opponent policy
and has its own graduation criterion.

| Stage | Opponent | Graduate when | Min episodes | Min epsilon |
|-------|----------|---------------|-------------|-------------|
| 1 | random_policy | win rate ≥ 70% × 2 consecutive evals | 300 | ≤ 0.80 |
| 2 | defensive_policy | win rate ≥ 68% × 2 consecutive evals | 300 | ≤ 0.60 |
| 3 | greedy_capture | win rate ≥ 40% × 2 consecutive evals | 300 | ≤ 0.45 |
| 4 | greedy_capture | no graduation — runs for remaining budget | — | — |

Stage 4 is a pure consolidation phase. Same opponent as Stage 3 but with a lower
learning rate and less frequent target updates, allowing the model to stabilise what
it learned in Stage 3. There is no self-play.

**Why two consecutive evals?** A single eval at 200 games has ±7% noise. Requiring
two consecutive evals above threshold reduces false graduations.

**Why a minimum epsilon?** Graduating while epsilon is still high means the model
hasn't actually learned the behaviour — it's partly winning by chance. The
`GRAD_MIN_EPS` thresholds open the graduation gate only once epsilon has decayed
into genuine exploitation territory.

**Evaluation:** every `eval_every` (default 50) episodes, 200 games are played
against the current stage's opponent with epsilon=0 and alternating seats. The result
is stored as `last_vs_opp` and displayed in the training log.

### Epsilon Schedule

```python
epsilon(step) = eps_start + min(step / eps_decay, 1.0) * (eps_end - eps_start)
```

Epsilon decays linearly from `eps_start` to `eps_end` over `eps_decay` gradient
steps. The step counter resets at each stage graduation (partial epsilon reset).

**Per-stage decay budgets:**
- Stage 1: 150,000 steps (`eps_decay_1`)
- Stage 2: 150,000 steps (`eps_decay_2`)
- Stage 3: 200,000 steps (`eps_decay_3`)
- Stage 4: uses Stage 3 decay schedule (no separate setting)

**Epsilon at graduation (partial reset):**

| Into Stage | eps_start |
|-----------|-----------|
| Stage 2 | 0.80 |
| Stage 3 | 0.60 |
| Stage 4 | 0.45 |

The reset is intentionally partial rather than back to 1.0. Resetting fully would
flood the buffer with random transitions right when the model needs stable gradients
from the harder opponent.

### Reward Shaping — All Values

All rewards listed below are in addition to the terminal reward of **+5.0 for win**
(defined in `env.py`). All other terminal cases return 0.0 from the environment.

#### Per-Move Rewards

These fire based on what happened during the current move's cascade.

**`REWARD_CLEAN_CAPTURE = 0.035`** — per enemy orb captured, only when exposure did
not increase. "Dirty" captures (that create new exposure) earn nothing; the exposure
penalties handle those. This specifically rewards efficient, safe aggression.

**`REWARD_CASCADE_OWN = 0.012`** — per net orb gained beyond the 1 placed this turn.
Any cascade that gains more than 1 orb for the agent earns a small bonus per excess
orb. Rewards triggering chain reactions rather than just placing.

#### Positional Rewards (delta-based unless noted)

These fire based on the change in board structure before vs after the move.

**`REWARD_SAFE_PRIME = 0.050`** *(phase-scaled)* — per net-new safe primed cell. A
"safe" primed cell is not adjacent to any enemy primed (not exposed). Stepping stone
toward chains. Phase-scaled up to 3× in the early game.

**`REWARD_ANY_PRIME = 0.001`** *(phase-scaled)* — per net-new primed cell regardless
of safety. Provides a minimal gradient signal for priming even unsafe cells. Safe
primes earn `REWARD_SAFE_PRIME + REWARD_ANY_PRIME` combined.

**`REWARD_CHAIN = 0.060`** *(full current score when chain grows)* — when the number
of my primed-to-primed adjacent pairs increases, the reward is `0.060 × chain_pairs`
(capped at `CHAIN_REWARD_CAP = 5` pairs). Longer chains pay more than shorter ones:
growing from 3 pairs to 4 pays `0.060 × 4 = 0.240`, not just `0.060 × 1`.

**`REWARD_SAFE_ATTACK = 0.055`** — per net-new primed cell that is adjacent to an
unprimed enemy cell and is not itself exposed. Rewards establishing pressure on
unprimed territory without leaving a primed cell at risk of being cascaded.

**`REWARD_ATTACK_CHAIN = 0.040`** *(full current score when attacking chain grows)* —
same "full current score" logic as `REWARD_CHAIN`, but only counting primed pairs
where at least one cell is adjacent to any enemy cell. Stacks on top of `REWARD_CHAIN`
for the same pairs — the combined signal for an attacking chain is `0.100` per pair
(capped). Kept lower than CHAIN to avoid double-counting.

**`REWARD_CORNER_CONTROL = 0.015`** *(phase-scaled)* — per newly claimed corner cell
(any orb count). Corners have critical mass 2, so a single orb primes them. Small
early-game reward for establishing corner foothold.

**`REWARD_CORNER_EXECUTE = 0.080`** *(phase-scaled)* — fires when a corner that was
primed and had an adjacent owned cell pre-move (the "setup" state) has now fired in
the post-cascade. Detected by checking whether the corner's count is still `cm−1`
post-cascade. Rewards the complete corner strategy: claim → prime → adjacent cell →
blast.

**`REWARD_EDGE_CHAIN_BONUS = 0.045`** — per net-new chain pair where at least one of
the two cells is on the board edge. Edge cells have lower critical mass than interior
(3 vs 4), so edge chains are more efficient. Encourages perimeter control.

**`REWARD_HUB_PRIME = 0.060`** — per net-new primed cell with ≥3 primed neighbours.
Hub primes are dense cluster junctions: when they fire, they simultaneously trigger 3
chain reactions. Requires ≥3 primed neighbours to exclude simple linear chains (an
interior cell needs 3 of 4 neighbours primed; edge cells need all 3; corner cells
can never qualify).

#### Tactical Execution Reward

**`REWARD_L_OVERLOAD_EXECUTE = 0.15`** — per enemy primed cell destroyed when an
L-overload configuration executes. An L-overload config is 3 of my primed cells in
a 2×2 block, with the 4th cell (any owner, at ≥`cm−2` orbs) adjacent to at least
one enemy primed. When this config exists pre-move and the cascade destroys enemy
primed cells post-move, the reward fires per enemy primed destroyed.

We reward execution rather than setup because the 4th cell in the L-config is
dangerous (it's adjacent to enemy primed — exposure). The correct play is to fire
immediately, not to sit and hold the position.

#### Phase Boost

```python
phase_boost = max(1.0, 3.0 - total_orbs / 10.0)
```

Applied to: `REWARD_SAFE_PRIME`, `REWARD_ANY_PRIME`, `REWARD_CORNER_CONTROL`,
`REWARD_CORNER_EXECUTE`.

At game start (0 orbs on board): `phase_boost = 3.0`. Fades to `1.0` at ~20 total
orbs. Incentivises corner and priming strategy in the first few moves before the
board fills up and tactical play dominates.

#### Exposure Resolution Reward (Type 2)

**`REWARD_EXPOSURE_GAIN = 0.10`** — per enemy primed destroyed when the force-greedy
override fires (exposure existed at start of turn + big cluster) AND exposure reduces
after the move. Only fires when the override actually triggered — it rewards the
model for being in a position where the forced fire resolves the threat.

#### Penalties

**`PENALTY_WASTED_PLACEMENT = 0.20`** — fired **pre-step** when a non-priming orb
is placed adjacent to an enemy primed cell. This is the worst move possible: you give
the enemy a free cascade target without gaining any defensive or offensive value.

Does NOT fire if the placement primes the cell (`new_count == cm−1`) or explodes it
(`new_count >= cm`) — those are intentional plays that happen to land near enemy primed.

**`PENALTY_SIM_CASCADE_LOSS = 0.15`** — per orb lost in simulated enemy fire. After
the move resolves, for each of the agent's exposed primed cells, each threatening
enemy primed cell is simulated firing, and the orb loss is counted. Penalised at
`0.15` per orb lost, hard-capped at `SIM_CASCADE_MAX_PENALTY = 0.45` total per step.

The cap prevents a heavily-exposed position from generating a `−3.0+` penalty that
drowns all positive signals and teaches the model to never approach the enemy.

**`PENALTY_EXPOSURE_MOVE = 0.25`** — pushed directly to the replay buffer for each
action blocked by the risk filter. This gives blocked risky moves a direct negative
gradient even though they were never actually played. Without this, blocked actions
receive zero gradient and their Q-values never decrease.

**`PENALTY_UNPRIMED_CELLS = 0.030`** — per owned unprimed cell above 5, fires when
`total_orbs > 10`. An unprimed cell (one that isn't at `cm−1`) contributes nothing to
chains or attacks. If you own more than 5 of them, you're spreading too thin.
Penalises scatter-and-hold: owning many single-orb cells is worse than owning fewer
primed ones.

#### Reward Summary Table

| Constant | Value | Type | Trigger |
|----------|-------|------|---------|
| Terminal win | +5.000 | terminal | Win the game |
| REWARD_CLEAN_CAPTURE | +0.035/orb | per-move | Enemy orbs captured, no new exposure |
| REWARD_CASCADE_OWN | +0.012/orb | per-move | Net orbs gained beyond 1 placed |
| REWARD_SAFE_PRIME | +0.050/cell | positional Δ | New safe primed cells (×phase_boost) |
| REWARD_ANY_PRIME | +0.001/cell | positional Δ | Any new primed cell (×phase_boost) |
| REWARD_CHAIN | +0.060×pairs | positional full | Chain grew (capped at 5 pairs) |
| REWARD_SAFE_ATTACK | +0.055/cell | positional Δ | New safe attacking primed cells |
| REWARD_ATTACK_CHAIN | +0.040×pairs | positional full | Attacking chain grew (capped at 5) |
| REWARD_CORNER_CONTROL | +0.015/cell | positional Δ | Newly claimed corner (×phase_boost) |
| REWARD_CORNER_EXECUTE | +0.080/corner | event | Corner fired from setup (×phase_boost) |
| REWARD_EDGE_CHAIN_BONUS | +0.045/pair | positional Δ | New edge chain pairs |
| REWARD_HUB_PRIME | +0.060/cell | positional Δ | New hub primed (≥3 primed neighbours) |
| REWARD_L_OVERLOAD_EXECUTE | +0.150/enemy | event | L-config fired, enemy primed destroyed |
| REWARD_EXPOSURE_GAIN | +0.10/enemy | event | Force-greedy resolves exposure |
| PENALTY_WASTED_PLACEMENT | −0.200 | pre-step | Non-priming orb next to enemy primed |
| PENALTY_SIM_CASCADE_LOSS | −0.15/orb | post-step | Orbs lost in simulated enemy fire |
| SIM_CASCADE_MAX_PENALTY | −0.45 cap | cap | Max Type 1 penalty per step |
| PENALTY_EXPOSURE_MOVE | −0.250 | buffer push | Each action blocked by risk filter |
| PENALTY_UNPRIMED_CELLS | −0.030/excess | ongoing | Owned unprimed > 5 (midgame only) |

### Positional Snapshot

`_positional_snapshot(game, player)` is called before each agent move and returns a
dict capturing the board state. `compute_shaping` uses before/after snapshots to
compute deltas.

| Field | Description |
|-------|-------------|
| `all_primes` | Total primed cells owned by player |
| `safe_primes` | My primed cells not adjacent to any enemy primed |
| `chain_pairs` | Adjacent my-primed pairs (`// 2`) |
| `safe_attack` | My primed adjacent to unprimed enemy, not exposed |
| `attack_chain_score` | `sum(REWARD_ATTACK_CHAIN/2)` over attacking chain pairs (both directions) |
| `corner_orbs` | Corners owned by player (any count) |
| `corner_setup_cells` | Frozenset of corners that are primed AND have ≥1 adjacent owned cell |
| `edge_chain_pairs` | Chain pairs where ≥1 cell is on board edge |
| `hub_primes` | Primed cells with ≥3 primed neighbours |
| `l_overload` | Count of 2×2 configs: 3 my primed + 4th cell at ≥cm−2, adjacent to enemy primed |
| `enemy_primed_count` | Total enemy primed cells (used for L-overload execution detection) |

When adding new positional rewards: add a field here AND in `return {}` AND the delta
in `compute_shaping` AND the panel row in `debug_watch.py`.

### Hardcoded Overrides

#### Force-Greedy Override (all stages)

Applied at the start of every agent turn in `_run_episode_vs_fixed`:

```python
force_greedy = (
    count_exposure(env.game, player) > 0
    and _has_big_cluster(env.game, player)
)

if force_greedy:
    r_act, c_act = greedy_capture_policy(env.game, player)
```

Conditions: exposure (my primed adjacent to enemy primed) exists AND a mixed primed
component of ≥4 cells exists that contains at least one of my cells.

**Why hardcode this?** When a large chain cluster is live, failing to fire is almost
always catastrophically bad. The model should learn to fire anyway, but hardcoding
ensures it happens during training so the reward signal from firing (positive outcome)
and not firing (orb loss, exposure penalty) is clean and unambiguous. This also
dramatically reduces games where the model passively loses to a chain it could have
pre-empted.

#### Corner Opening Hardcode (Stages 1–2 only)

For the agent's first 3 turns in Stages 1 and 2 (`stage <= 2`, `agent_turn_count < 3`),
a deterministic opening is injected:

- **Turn 0:** place on a "clean" corner (corner and all its neighbours are enemy-free). Corner becomes primed.
- **Turn 1:** place on an adjacent edge cell. Sets up the blast: agent owns both the corner and an adjacent cell.
- **Turn 2:** place on the corner again. Corner fires, cascades into the adjacent cell. `REWARD_CORNER_EXECUTE` fires.

Falls through silently to `_exec_safe_action` if the plan can't proceed at any step
(corner captured, no valid adjacent cell, corner lost).

Disabled from Stage 3 onward — by then the model has internalised the corner opening
from thousands of repetitions.

### Risk Filter

**`_move_risk_score(game, player, r, c)` in `game_analysis.py`**

Active only when `total_orbs >= 28` (midgame). Returns `0.0` unconditionally before
that threshold, and `0.0` if the agent's move would win the game outright.

Three-snapshot evaluation:

1. **Before our move** — record my primed and unprimed cell counts as baseline.
2. **After our cascade** — compute `our_gain = 1.0 × primed_gained + 0.2 × unprimed_gained`.
3. **After enemy's single best greedy reply** — compute `our_loss = 1.0 × primed_lost + 0.2 × unprimed_lost`.

Returns `max(0.0, our_loss - our_gain)`. Zero means the trade is safe (or the filter
is inactive). Any positive value means our loss exceeds our gain — block the move.

**Primed cells are weighted 5× more than unprimed** (1.0 vs 0.2) because primed cells
are near-term chain participants while unprimed cells are passive territory.

**`_exec_safe_action` in `train.py`** wraps the risk filter:

1. Agent selects an action.
2. Compute risk score. If 0.0, return immediately (safe).
3. If risky: simulate the move, encode bad next_state, push penalty transition
   `(state, action, −PENALTY_EXPOSURE_MOVE, bad_next_state, False, bad_mask)` to teach
   Q-values directly that this action is bad. Remove from `remaining` mask.
4. Retry until a safe action is found, or all moves are exhausted.
5. If all moves are net-negative, pick the least-risky one (minimum risk score).

**Why push penalty transitions for blocked moves?** Without gradient feedback,
blocked actions have unchanged Q-values and will keep being selected. The penalty
push creates a direct negative update: the Q-value for that action decreases,
improving future action selection without the agent ever actually playing the bad move.

### Sliding-Window Replay Buffer

At each graduation, old transitions are evicted to remove stale gradient signal:

```
S1→S2: record Stage 1 buffer size N1. No eviction. (S2 trains on S1+S2 data)
S2→S3: evict first N1 entries. Keep only Stage 2. (S3 trains on S2+S3 data)
S3→S4: evict first N2 entries. Keep only Stage 3. (S4 trains on S3+S4 data)
S4+  : no eviction — same greedy opponent; all data remains valid.
```

**Why evict S1 at S3?** Stage 1 data was collected against a random opponent with no
exposure-awareness. If it stays in the buffer through Stage 3, the model gets
contradictory gradients: "this position is fine" (from Stage 1 when the random
opponent never punished it) vs "this position is dangerous" (from Stage 3 when the
greedy opponent immediately exploits it).

### Hyperparameters

All CLI defaults for `train.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--rows` | 6 | Board rows |
| `--cols` | 6 | Board cols |
| `--channels` | 64 | Hidden conv channels |
| `--episodes` | 10000 | Total training episodes |
| `--lr` | 5e-5 | Adam LR (Stage 1 only; overridden per stage) |
| `--gamma` | 0.97 | Discount factor |
| `--batch_size` | 64 | SGD mini-batch size |
| `--buffer_size` | 500,000 | Replay buffer capacity |
| `--target_update` | 1000 | Steps between target net copies |
| `--eps_start` | 1.0 | Stage 1 initial epsilon |
| `--eps_end` | 0.05 | Final epsilon (all stages) |
| `--eps_decay_1` | 150,000 | Steps for Stage 1 epsilon decay |
| `--eps_decay_2` | 150,000 | Steps for Stage 2 epsilon decay |
| `--eps_decay_3` | 200,000 | Steps for Stage 3/4 epsilon decay |
| `--log_every` | 50 | Episodes between log lines |
| `--eval_every` | 50 | Episodes between evaluations |
| `--save_every` | 500 | Episodes between checkpoint saves |
| `--resume` | None | Resume from checkpoint path |

**Per-stage overrides (hardcoded constants, not CLI):**

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|--|---------|---------|---------|---------|
| LR | 4e-5 | 2e-5 | 2e-5 | 1e-5 |
| Target update freq | 1000 | 1000 | 1500 | 2000 |
| Epsilon reset | 1.0 | 0.80 | 0.60 | 0.45 |

**Why gamma=0.97?** A shorter effective horizon (~33 steps to discount to ~1/e)
prevents Q-value inflation from far-future rewards, which on a 6×6 board often
never materialise anyway (games typically end in 20–50 plies).

---

## Debug Visualiser — debug_watch.py

A step-by-step reward breakdown tool that shows every shaping component for each move
as it's played. Useful for diagnosing why the model makes specific decisions.

**Usage:**

```bash
# Watch trained agent as P0 vs greedy, with training overrides active
python rl/debug_watch.py --ckpt0 rl/final/final.pt --policy1 greedy --force_greedy --filter_moves

# Two policies (no model)
python rl/debug_watch.py --policy0 greedy --policy1 defensive

# Agent vs agent
python rl/debug_watch.py --ckpt0 rl/final/final.pt --ckpt1 rl/checkpoint5500/final.pt
```

**Controls:**

| Key | Action |
|-----|--------|
| Space / Enter | Advance one move |
| U | Undo last move (full game state restore) |
| R | Restart game |
| Esc | Quit |

**Reward panel:** right side overlay shows every component (capture, cascade,
safe_prime, chain, safe_atk, atk_chain, corner_ctrl, corner_execute, edge_chain,
hub_prime, l_overload_exec, sim_casc, exp_gain, unprimed, wasted, TERMINAL) with
colour-coded bars and a running cumulative total per player.

**Flags:**

- `--force_greedy`: mirrors the training override (fire when big cluster + exposure)
- `--filter_moves`: mirrors the training risk filter (block net-negative trades)
- `--epsilon`: exploration rate for agent seats (default 0.0 = fully greedy)

**Important:** `debug_watch.py` computes reward breakdowns using a separate `sim`
copy of the game state to preserve the actual `game` for rendering. The `breakdown`
function is independent of `compute_shaping` in `train.py` — they share the same
constants (imported directly) and the same logic, but are implemented separately.

---

## Arena Evaluator — arena.py

Headless N-game evaluation between two checkpoints. Alternates seats every game so
neither model benefits from going first.

**Usage:**

```bash
python rl/arena.py \
    --ckpt0 rl/final/final.pt \
    --ckpt1 rl/checkpoint5500/final.pt \
    --games 400 \
    --epsilon 0.05 \
    --force_greedy --filter_moves --verbose
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--games` | 400 | Total games (should be even for fair seat split) |
| `--epsilon` | 0.05 | Random-move probability (prevents deterministic identical games) |
| `--force_greedy` | off | Mirror training override |
| `--filter_moves` | off | Mirror training risk filter |
| `--verbose` | off | Print per-game results |

**Important:** using `--epsilon 0.0` with identical models produces identical games
every time (both play deterministically from the same start position). Always use
`epsilon > 0` for meaningful statistics. 0.05 is recommended.

**Output format:**

```
Model          Wins  As-P0  As-P1  Win%
final.pt        218    112    106  54.5%
ep07000.pt      182     88     94  45.5%
```

---

## Passive Viewer — watch.py

Watches any combination of agents and policies play with the full pygame renderer.
Unlike `debug_watch.py`, there is no reward panel — this is purely for watching the
game play out.

**Usage:**

```bash
# Agent vs greedy, with 600ms delay between moves
python rl/watch.py --ckpt0 rl/final/final.pt --policy1 greedy --delay 600

# Two agents head-to-head
python rl/watch.py --ckpt0 rl/final/final.pt --ckpt1 rl/checkpoint5500/final.pt

# Human (seat 1) vs AI (seat 0)
python rl/watch.py --ckpt0 rl/final/final.pt --human 1
```

**Controls:** Space=pause, R=restart, Esc=quit. Auto-restarts after each game.

---

## Checkpoints

### final/final.pt — Primary checkpoint

**The best checkpoint.** Trained for 10,000 episodes on a 6×6 board, completed all
4 curriculum stages.

- **Stage graduated:** Stage 4 (greedy consolidation — no further graduation)
- **Total gradient steps:** 306,847
- **Training time:** ~5,157 seconds (~86 minutes), CPU only
- **Win rates (200 games, alternating seats):** 95% vs random, 81.5% vs defensive, 57% vs greedy

Arena results vs earlier checkpoints: `final.pt` wins 55–68% against every prior
checkpoint from the same run, establishing clear improvement across the full training
run.

### checkpoint5500/final.pt — Secondary checkpoint

An earlier training run with slightly different hyperparameters and lower graduation
thresholds (random ≥60%, defensive ≥65% vs the final run's 70%/68%). This run
experienced a severe Stage 2 crash (ep 3200–3450, loss peaked at 0.90, win rate
dropped to 13% vs defensive) caused by insufficient `target_update` frequency (1000)
for the Stage 3 difficulty jump. The model recovered and graduated but the crash
permanently damaged final Stage 3 performance.

- **Notable:** stronger vs defensive than the final run (~83% vs ~81.5%), but weaker vs greedy
- **Kept as:** reference for arena comparison; useful for testing whether model improvements are real

---

## Training Run History

### Final Run (rl/final/logs.txt)

Complete 10,000-episode run with the polished curriculum and all reward fixes applied.

**Stage 1 — vs random (ep 1–1400):**

Started at eps=1.0, decayed at 150k steps. Early evals showed healthy 64–77% win
rates. A notable dip at ep 900–950 (64–66%) during the loss spike phase (loss reached
0.69 at ep 1000) before recovering. Graduated at ep 1400 with 77.0% win rate,
eps=0.720. Buffer at graduation: 44,234 transitions.

**Stage 2 — vs defensive (ep 1400–2750):**

Transitioned smoothly with eps reset to 0.80. Win rate jumped immediately to 37.5%
at ep 1450 (model retained Stage 1 knowledge). Crossed 70% frequently from ep 1750
onward and graduated at ep 2750 with 81.0% win rate, eps=0.592. Buffer: evicted
44,234 Stage 1 transitions, kept 41,592 Stage 2 transitions.

**Stage 3 — vs greedy (ep 2750–4600):**

Typical transition shock: 0.0% at ep 2750 (eval immediately after graduation before
new experience). Quickly recovered to 30–41% range. Graduation required ≥40% × 2
evals and eps ≤ 0.445. Graduated at ep 4600 with 55.0% win rate, eps=0.445. Buffer:
evicted 41,592 Stage 2 transitions, kept 56,540 Stage 3 transitions.

**Stage 4 — greedy consolidation (ep 4600–10000):**

Win rate stabilised in the 50–60% range (greedy is a strong opponent; 57% at terminal
evaluation is a genuine achievement). Loss climbed slowly from 0.116 to 0.144 as
epsilon decayed — a sign of deeper exploitation rather than divergence. Epsilon at
ep 10000: 0.121 (well above eps_end=0.05, meaning more training budget could still
improve it). Training time: 5,157 seconds total (~86 minutes on CPU).

### Checkpoint5500 Run (rl/checkpoint5500/logs.txt)

Earlier run with different settings. Key events:

- **Stage 1 → 2:** graduated ep 1100 (83.0%, eps=0.780) — much faster due to lower 60% threshold
- **Stage 2 crash:** ep 3200–3450, loss spiked from 0.26 to 0.90; win rate dropped to 13% then 20%
  - Cause: `target_update=1000` was too frequent for Stage 3 difficulty; TD targets oscillated wildly
  - Recovery: model self-corrected by ep 3600 (43.5%), graduated ep 3700
- **Stage 2 → 3:** graduated ep 3700 (80.0%, eps=0.400)
- **Stage 3 performance:** win rates mostly in 40–50% range, less stable than the final run

The crash event is the primary reason the final run uses `STAGE_TARGET_UPDATE = {3: 1500, 4: 2000}`.

---

## Known Issues and Design Decisions

### Resolved Issues

**Q-value divergence (target_update=300):** early runs used 300-step target updates,
causing monotonically increasing loss. Fixed to 1000+ and per-stage escalation.

**Stage 1 too fast (eps_decay_1=80k):** graduating before sufficient pattern coverage
meant the model didn't generalise. Fixed to 150k steps.

**Blocked action Q-values:** risky actions blocked by `_exec_safe_action` received no
gradient, so their Q-values never decreased and they kept being selected. Fixed by
pushing `−PENALTY_EXPOSURE_MOVE` penalty transitions directly to the buffer.

**Unbounded chain rewards:** "full current score" with no cap caused Q-value inflation
in Stage 2 (model learned to obsess over chains and nothing else). Fixed with
`CHAIN_REWARD_CAP = 5`.

**sim_cascade overflow (−3.0+ per step):** a single heavily-exposed position could
generate massive penalties, drowning all positive signals. Fixed with
`SIM_CASCADE_MAX_PENALTY = 0.45`.

**L-build misfiring:** an earlier "L-overload build" reward (for setting up the
configuration) fired from cascade-created accidental primes and re-fired every step
the config existed. Removed entirely. Only execution is rewarded now.

**Corner setup confusion:** rewarding the corner setup (corner primed + adjacent
owned) was wrong — that configuration is dangerous (exposure: the 4th cell adjacent
to enemy primed). Changed to `REWARD_CORNER_EXECUTE`: reward the blast, not the
preparation.

**Stage 3 graduation gate too late (GRAD_MIN_EPS[3]=0.30):** opened the graduation
gate inside the crash zone where epsilon was too low and loss was spiking. Fixed to
0.45 — gate opens earlier, during post-crash recovery phase.

**PENALTY_UNRESOLVED_EXPOSURE (Type 2) removed:** a flat per-chain-size exposure
penalty for not resolving exposure each turn created noisy gradients (same exposure
penalised every turn regardless of whether the model could do anything about it).
Replaced by the risk filter (blocks net-negative trades pre-move) and the Type 1
simulation penalty (penalises remaining exposure post-move via cascade sim).

**`_move_risk_score` summing all threats independently:** old logic ran each enemy
primed cell firing separately and summed penalties — wildly over-counted danger by
assuming all threats fire simultaneously. Replaced with a single greedy-reply
simulation (one enemy move) and a relative gain/loss comparison.

**`parse_args` missing `--resume`:** `train.py` exited silently with `AttributeError`
when resuming from checkpoint. Fixed by adding the argument.

**phase_boost divisor:** early versions used `/4.0` (faded to 1× at ~8 total orbs,
too fast). Changed to `/10.0` (fades to 1× at ~20 orbs), giving the early-game boost
enough time to actually influence the first 5–8 turns.

### Active Issues

**Stage 2 crash with sliding-window buffer:** when Stage 1 data (random opponent,
no exposure punishment) remains in the buffer during Stage 2, the model gets
contradictory Q-targets as the defensive opponent starts punishing exposure. The
final run mitigated this by widening `STAGE_TARGET_UPDATE` and lowering Stage 2 LR,
but no structural fix exists. Watch for `avg_loss > 0.8` in Stage 2 as a warning sign.
