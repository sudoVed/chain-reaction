# Chain Reaction — Project Context for New Sessions

## What this project is

A Python implementation of the board game **Chain Reaction** (2-player, 6×6 default grid)
with a DQN reinforcement learning agent trained via a 4-stage curriculum. The game has a
pygame renderer for human/agent play, and a headless RL environment for training.

---

## File map

```
ChainReaction/
├── game.py              Core game logic (no pygame). Wave-based cascade, Cell, Game.
├── main.py              Pygame entry point. Human vs human / human vs agent.
├── renderer.py          Pygame renderer. Animated flying orbs, wave sync.
├── constants.py         Visual/timing constants for the renderer.
├── CLAUDE.md            This file.
├── rl/
│   ├── env.py           Gym-style wrapper around game.py. State encoding, cascade sim.
│   ├── model.py         Spatial CNN Q-network (3 conv blocks + 1×1 head).
│   ├── agent.py         DQNAgent: replay buffer, epsilon-greedy, target net, train_step.
│   ├── policies.py      Three fixed opponent policies: random, greedy_capture, defensive.
│   ├── game_analysis.py Pure board analysis helpers (shared across train/arena/debug_watch).
│   ├── train.py         Full training loop: reward shaping, curriculum, eval, checkpoints.
│   ├── debug_watch.py   Per-move reward breakdown visualiser (pygame) with undo support.
│   ├── arena.py         Headless model-vs-model evaluation. Win/draw stats, seat alternation.
│   ├── watch.py         Watch a trained agent play (pygame display).
│   ├── trainold2.py     Backup of train.py before the big reward rewrite.
│   ├── trainold3.py     Backup of train.py before the game_analysis.py refactor.
│   └── NOTES.md         Ongoing training notes.
```

---

## Game rules — CRITICAL (was buggy, now fixed)

- Players alternate placing one orb on any empty cell or cell they own.
- Each cell has a **critical mass** equal to its neighbour count: corner=2, edge=3, interior=4.
- When a cell reaches critical mass it **explodes**: scatters one orb to each neighbour.
- **CRITICAL RULE:** any orb landing on any cell — empty OR enemy — immediately converts it
  to the attacker's colour, even if it does NOT subsequently explode. This was previously
  broken (only empty cells converted). All checkpoints before this fix are **invalid**.
- Explosions are processed wave-by-wave simultaneously.
- A player is eliminated when they own zero cells after any wave (once they've had their
  first turn). Last player standing wins.

---

## Neural network architecture

**File:** `rl/model.py`

Fully-convolutional DQN — works on any board size with the same weights.

```
Input:  (batch, 4, H, W)
Block1: Conv2d(4→32,  3×3, pad=1) + BN + ReLU
Block2: Conv2d(32→64, 3×3, pad=1) + BN + ReLU
Block3: Conv2d(64→64, 3×3, pad=1) + BN + ReLU
Head:   Conv2d(64→1,  1×1)
Output: flatten → (batch, H*W)  — one Q-value per cell
```

Default: 64 hidden channels (`--channels 64`).

---

## State encoding

**File:** `rl/env.py`

Always encoded from the perspective of the player about to move. Flat float32 array of
length `4 * rows * cols`, four stacked channels:

| Channel | Content                                                      |
|---------|--------------------------------------------------------------|
| 0       | my orb counts / 4.0                                          |
| 1       | enemy orb counts / 4.0                                       |
| 2       | critical_mass / 4.0  (static geometry; corner=0.5, etc.)    |
| 3       | primed map: +1.0 my primed, -1.0 enemy primed, 0.0 else     |

`+1` next to `-1` = exposure (danger). `+1` next to `+1` = your chain.

---

## Curriculum stages

**File:** `rl/train.py`

| Stage | Opponent         | Graduate when                              |
|-------|------------------|--------------------------------------------|
| 1     | random_policy    | win rate vs random    ≥ 70%  × 2 evals    |
| 2     | defensive_policy | win rate vs defensive ≥ 68%  × 2 evals    |
| 3     | greedy_capture   | win rate vs greedy    ≥ 40%  × 2 evals    |
| 4     | greedy_capture   | (no graduation — runs for remaining budget)|

Stage 4 is pure greedy — deepens and stabilises Stage 3 greedy-fighting skills. No
opponent mixing. `run_stage = min(stage, 3)` gates the policy lookup; Stage 4 overrides
directly to `greedy_capture_policy`.

Graduation only checked after `CURRICULUM_MIN_EPS = 300` stage episodes AND
`epsilon ≤ GRAD_MIN_EPS[stage]`.

```python
GRAD_RANDOM     = 0.70
GRAD_DEFENSIVE  = 0.68
GRAD_GREEDY     = 0.40
GRAD_MIN_EPS    = {1: 0.80, 2: 0.60, 3: 0.45}
```

Epsilon is **per-stage** — partial reset at each graduation (not to 1.0):

```python
STAGE_EPS_START = {2: 0.80, 3: 0.60, 4: 0.45}
```

Stage 1 always starts at `eps_start` (CLI default 1.0).

Epsilon decays:
- Stage 1: `eps_decay_1 = 150,000` steps  (slow: model needs diverse patterns before Stage 2)
- Stage 2: `eps_decay_2 = 150,000` steps
- Stage 3: `eps_decay_3 = 200,000` steps
- Stage 4: uses Stage 3 decay schedule (no separate arg)

---

## Per-stage hyperparameters

```python
STAGE_LR        = {1: 4e-5, 2: 2e-5, 3: 2e-5, 4: 1e-5}
STAGE_TARGET_UPDATE = {1: 1000, 2: 1000, 3: 1500, 4: 2000}
```

LR drops at each graduation to stabilise Bellman targets on harder opponents.
Target network update frequency rises in later stages to dampen crash severity when
epsilon decays into the exploitation zone.

---

## Sliding-window replay buffer

At each graduation the buffer is pruned to remove the oldest stage's data:

```
S1→S2: record Stage 1 buffer size N1. No eviction yet. (Stage 2 trains on S1+S2 data)
S2→S3: evict N1 oldest entries. Keep only Stage 2 transitions. (Stage 3 trains on S2+S3)
S3→S4: evict N2 oldest entries. Keep only Stage 3 transitions. (Stage 4 trains on S3+S4)
S4+  : no eviction — same greedy opponent, all data is still valid.
```

Rationale: Stage 1 data (random opponent, no exposure punished) teaches the wrong Q-values
for Stage 3+. Evicting two stages back is cheaper than a full cold start.

Buffer size default: **500,000** (`--buffer_size 500_000`).

---

## Opponent policies

**File:** `rl/policies.py`

**random_policy:** uniform random legal move.

**greedy_capture_policy:** priority: (1) own primed adj to enemy primed, (2) any own
primed, (3) random.

**defensive_policy:** avoids placing next to enemy primed. Priority: (1) safe own primed
adj to enemy primed (preemptive), (2) safe own primed, (3) any safe move, (4) dangerous
own primed (last resort), (5) any legal move.

---

## Replay buffer and agent

**File:** `rl/agent.py`

The `ReplayBuffer` is a circular deque of fixed capacity. Each entry is a 6-tuple:
`(state, action, reward, next_state, done, next_mask)`.

**Bellman target (1-step zero-sum):**
```
target = r + gamma * (1 - done) * (-max Q̂(s'))
```
`s'` is always from the **opponent's** perspective, so negate their best Q-value.

`max_norm = 1.0` gradient clipping in `train_step`.

---

## Episode runner (train.py)

**Function:** `_run_episode_vs_fixed` (Stages 1–4)

### Per-step flow:

```
BEFORE agent's move:
  exposure_before  = count_exposure(game, player)
  enemy_primed_pre = count_primed(game, enemy)
  pos_before       = _positional_snapshot(game, player)
  counts_before    = game.orb_counts()
  cell_snapshot    = {(r,c): count} for player's cells

IF force_greedy AND exposure_before > 0 AND _has_big_cluster(game, player):
  → action = greedy_capture_policy (hardcoded fire)
ELIF agent_turn_count < 3 AND stage <= 2:
  → try _corner_hardcode_action(...); fall through to _exec_safe_action if it returns None
    or if the hardcoded cell is not in the valid mask
ELSE:
  → action, r_act, c_act = _exec_safe_action(...)

wasted = wasted_placement_penalty(game, player, r_act, c_act)

env.step(action) → next_state, reward, done

IF done:
  r_t = reward + wasted
  push(state, action, r_t, next_state, done, mask)
ELSE:
  exposure_after = count_exposure(game, player)
  r_t = compute_shaping(...) + wasted
  if exposure_before > 0 and exposure_after < exposure_before and force_greedy fired:
    r_t += REWARD_EXPOSURE_GAIN * primed_destroyed
  push(state, action, r_t, next_state, done, mask)

agent.train_step()
```

---

## Risk filter: _exec_safe_action and _move_risk_score

**File:** `rl/game_analysis.py`

**`_move_risk_score(game, player, r, c)`** — relative risk filter, active only when
`total_orbs >= 28` (midgame). Returns `max(0, our_loss - our_gain)`. Three snapshots:

1. **Before our move** — record player's primed and unprimed cell counts.
2. **After our cascade** — compute `our_gain = 1.0*primed_gained + 0.2*unprimed_gained`.
   If our move already wins the game, return `0.0` (no enemy reply).
3. **After enemy's single greedy reply** (`greedy_capture_policy`) — compute
   `our_loss = 1.0*primed_lost + 0.2*unprimed_lost`.

Returns `max(0.0, our_loss - our_gain)`. A value of `0.0` means the trade is safe or
the filter is inactive; any positive value means our loss exceeds our gain.

**`_exec_safe_action(env, player, mask, agent, state, epsilon)`** — wraps action selection:
1. Agent selects an action.
2. Compute `risk = _move_risk_score(...)`.
3. If `risk == 0.0`: return the action (safe or filter inactive).
4. Else: simulate the risky move, encode bad next_state, push penalty transition
   `(state, action, -PENALTY_EXPOSURE_MOVE, bad_next, False, bad_mask)` to teach the
   Q-network directly that risky moves are bad. Remove from `remaining` mask.
5. Retry until safe action found or all moves exhausted (falls back to least-risky).

---

## Reward shaping — current values

**File:** `rl/train.py` (all constants at top of file)

Terminal reward: **+5.0 for win** (set in `env.py`).

### Per-move rewards
```python
REWARD_CLEAN_CAPTURE    =  0.035   # per enemy orb captured when exposure doesn't increase
REWARD_CASCADE_OWN      =  0.012   # per net orb gained beyond the 1 placed (cascade payoff)
```

### Positional rewards

`REWARD_CHAIN` and `REWARD_ATTACK_CHAIN` use **"full current score when it grows"** logic:
growing a longer chain rewards more than a short one. Both are capped at `CHAIN_REWARD_CAP = 5`
pairs to prevent Q-value inflation.
All other positional rewards are true delta (net-new this turn only).

```python
REWARD_SAFE_PRIME       =  0.050   # per net-new safe primed cell (not adj to enemy primed)
                                   # Phase-scaled: 3× at game start, 1× after ~20 total orbs
REWARD_CHAIN            =  0.060   # full chain-pair score when chain grows (capped at 5)
REWARD_SAFE_ATTACK      =  0.055   # per net-new primed adj to UNPRIMED enemy (not exposed)
REWARD_ATTACK_CHAIN     =  0.040   # full attacking chain score when it grows (stacks with CHAIN)
REWARD_CORNER_CONTROL   =  0.015   # per newly claimed corner cell
                                   # Phase-scaled: 3× at game start, 1× after ~20 total orbs
REWARD_CORNER_EXECUTE   =  0.080   # corner blast executed: corner was primed+adjacent-owned
                                   # pre-move, post-cascade count is no longer cm-1
                                   # Phase-scaled: same boost as CORNER_CONTROL
REWARD_EDGE_CHAIN_BONUS =  0.045   # per net-new chain pair where >= 1 cell is on board edge
CHAIN_REWARD_CAP        =  5       # max chain pairs counted in full-score rewards
REWARD_ANY_PRIME        =  0.001   # per net-new primed cell regardless of safety (stepping stone)
                                   # Phase-scaled: same boost as SAFE_PRIME
```

### Tactical structure rewards (delta-based)
```python
REWARD_HUB_PRIME          =  0.060  # per net-new primed with >= 3 primed neighbours (dense cluster node)
                                    # Requires >= 3 to exclude simple linear chains
REWARD_L_OVERLOAD_EXECUTE =  0.15   # per enemy primed destroyed during L-overload execution
```

**Phase boost:** `phase_boost = max(1.0, 3.0 - total_orbs / 10.0)`
Applied to `REWARD_SAFE_PRIME`, `REWARD_ANY_PRIME`, `REWARD_CORNER_CONTROL`, and
`REWARD_CORNER_EXECUTE`. Early-game these rewards are up to 3× more valuable; they fade
to 1× at ~20 total orbs on the board.

**L-overload execution:** Config pre-move = 3 my primed in a 2×2 + 4th cell (any owner)
at count ≥ cm-2, adjacent to at least one enemy primed. Post-cascade: enemy primed count
decreased → reward fires per enemy primed destroyed. We reward **execution not setup**
because having the 4th cell adjacent to enemy primed is dangerous (exposure); the correct
play is to immediately fire, not hold the configuration.

**Hub prime:** requires ≥ 3 primed neighbours (not 2) to exclude linear chains. In a 6×6
board an interior cell needs 3 of its 4 neighbours primed. Edge cells (3 neighbours) need
all 3. Corner cells (2 neighbours) can never qualify.

**Corner strategy:** corner cell has cm=2, so 1 orb = primed. The sequence is:
1. Place 1 orb in corner → corner primed → `REWARD_CORNER_CONTROL` fires (phase-boosted)
2. Own any adjacent cell → setup ready, stored in `pos_before['corner_setup_cells']`
3. Place 2nd orb in corner → corner blasts → `REWARD_CORNER_EXECUTE` fires

### Exposure resolution reward
```python
REWARD_EXPOSURE_GAIN    =  0.10    # per enemy primed destroyed when force-greedy resolves exposure
```
Fires in runner when force_greedy fires (big cluster AND exposure) and exposure reduces.

### Penalties
```python
PENALTY_WASTED_PLACEMENT    =  0.20   # placing non-priming orb next to enemy primed
PENALTY_SIM_CASCADE_LOSS    =  0.15   # per orb lost in simulated enemy fire (Type 1)
SIM_CASCADE_MAX_PENALTY     =  0.45   # hard cap on Type 1 total per step (prevents -3.0+ swamps)
PENALTY_EXPOSURE_MOVE       =  0.25   # pushed to buffer for each action blocked by risk filter
PENALTY_UNPRIMED_CELLS      =  0.030  # per owned unprimed cell above 5, fires when total_orbs > 10
```

**Type 1** (`PENALTY_SIM_CASCADE_LOSS`): fires inside `compute_shaping`. For each exposed
primed cell, simulate each threatening enemy primed cell firing, count orb loss, multiply
by coefficient. Hard-capped at `SIM_CASCADE_MAX_PENALTY = 0.45`.

Note: Type 2 (`PENALTY_UNRESOLVED_EXPOSURE`) was removed. Ignoring exposure is now
handled implicitly — the risk filter blocks net-negative trades pre-move, and the Type 1
penalty penalises any remaining exposure post-move via simulation.

**Wasted placement**: fires pre-step. Non-priming orb placed next to enemy primed = free
capture for them. Does NOT fire if placement primes or explodes the cell (intentional play).

**Unprimed cell penalty**: fires in `compute_shaping` when `total_orbs > 10`. Counts
player's owned cells with < cm-1 orbs (not primed). If `owned_unprimed > 5`, penalises
`PENALTY_UNPRIMED_CELLS * (owned_unprimed - 5)` per turn. Provides continuous pressure to
consolidate scattered single-orb cells into primed positions.

---

## Force-greedy override

In ALL stages (in `_run_episode_vs_fixed`):
```
if exposure_before > 0 AND _has_big_cluster(game, player):
    action = greedy_capture_policy(game, player)
```

`_has_big_cluster`: returns True if any mixed primed component (both players) has ≥ 4 cells
AND at least 1 of those cells is owned by the current player.

This hardcodes the decision to fire when a large chain cluster is present and the player
has exposure. The model should learn to fire anyway, but this ensures it happens during
training so the reward signal is clean.

---

## Corner opening hardcode

**Functions:** `_get_clean_corner`, `_corner_hardcode_action` in `rl/train.py`

Active for the agent's first **3 turns** in **Stages 1 and 2 only** (`stage <= 2`).
Injects a deterministic opening so the model sees the corner strategy from the very first
episode rather than discovering it by chance.

Turn sequence:
- **Turn 0** — place on a "clean" corner (corner + all its neighbours must be enemy-free).
  Corner becomes primed (count=1, cm=2).
- **Turn 1** — place on an adjacent edge cell (prefers edge over interior; any non-enemy cell).
  Sets up the blast: we now own the corner and an adjacent cell.
- **Turn 2** — place on the corner again → corner explodes, cascades into adjacent cell.
  `REWARD_CORNER_EXECUTE` fires here.

Falls through to `_exec_safe_action` silently if: the plan fails at any step (corner
captured, no valid adjacent cell, corner not still primed), or if the hardcoded action
is not in the valid mask.

Disabled from Stage 3 onward — by then the model has internalised the opening.

---

## Training hyperparameters (CLI defaults)

```
--rows 6  --cols 6  --channels 64  --episodes 10000
--lr 5e-5  --gamma 0.97  --batch_size 64  --buffer_size 500_000
--target_update 1000  --eps_start 1.0  --eps_end 0.05
--eps_decay_1 150_000  --eps_decay_2 150_000  --eps_decay_3 200_000
--log_every 50  --eval_every 50  --save_every 500
--resume PATH   (optional: resume from checkpoint)
```

Note: `--lr`, `--target_update` are overridden by `STAGE_LR` and `STAGE_TARGET_UPDATE`
at each graduation. The CLI values serve as Stage 1 fallbacks only.

**gamma=0.97** — shorter effective horizon, prevents Q-value inflation from far-future rewards.

---

## _positional_snapshot fields

**Function:** `_positional_snapshot(game, player)` in `train.py`

Computed pre-move, stored in `pos_before`, used by `compute_shaping` for deltas.

```python
{
  'all_primes':         int,        # total primed cells owned by player
  'safe_primes':        int,        # my primed cells not adj to any enemy primed
  'chain_pairs':        int,        # count of adjacent my-primed pairs (// 2)
  'safe_attack':        int,        # my primed adj to UNPRIMED enemy, not exposed
  'attack_chain_score': float,      # sum(REWARD_ATTACK_CHAIN/2 per directed attacking pair)
  'corner_orbs':        int,        # count of corners owned by player (any orb count)
  'corner_setup_cells': frozenset,  # corners that are primed AND have adjacent owned cell
  'edge_chain_pairs':   int,        # chain pairs where >= 1 cell is on board edge
  'hub_primes':         int,        # primed cells with >= 3 primed neighbours
  'l_overload':         int,        # count of 2x2 configs ready to execute (3 primed + 4th at cm-2 adj enemy primed)
  'enemy_primed_count': int,        # total enemy primed cells (for l_overload execution detection)
}
```

If you add new positional rewards: add field to `_positional_snapshot` AND its `return {}`
dict AND the delta computation to `compute_shaping` AND the breakdown to `debug_watch.py`.

`debug_watch.py` uses `.get('field', 0)` for snapshot fields for forward-compatibility.

Note: `PENALTY_UNPRIMED_CELLS` does NOT use a snapshot field — it's computed directly
in `compute_shaping` post-move from the live board state.

---

## debug_watch.py controls

```
Space / Enter  : advance one move
U              : undo last move (full game state restore)
R              : restart game
Esc            : quit
```

Panel shows every shaping component (r_t) for the last move, including the unprimed cell
penalty. Does NOT show the opponent's response. Total shown is approximately what
`compute_shaping` returns, not the final training signal (which also includes the wasted
placement penalty).

```
python rl/debug_watch.py --ckpt0 rl/checkpoints/final.pt  # watch trained agent as P0
                         --policy1 greedy                  # opponent policy
                         --force_greedy                    # mirror training override
                         --filter_moves                    # mirror risk filter
```

---

## arena.py — model vs model evaluation

**File:** `rl/arena.py`

Headless N-game evaluation between two checkpoints. Alternates seats every game.

```
python rl/arena.py --ckpt0 rl/checkpoints/ep05000.pt \
                   --ckpt1 rl/checkpoints/final.pt   \
                   --games 400 --epsilon 0.05         \
                   --force_greedy --filter_moves --verbose
```

Key flags:
- `--epsilon 0.05`  — inject randomness so games from identical start positions diverge.
  Use `--epsilon 0.0` for pure deployment-style deterministic play.
- `--force_greedy`  — mirrors training override (fire when big cluster + exposure).
- `--filter_moves`  — mirrors training risk filter (block moves where our_loss > our_gain, midgame only).
- `--verbose`       — print per-game winner + ply count.

Output: summary table with Wins / As-P0 / As-P1 / Win% per model.

**Important:** epsilon=0 on identical start positions produces identical games every time
(both models play identically → deterministic). Always use epsilon > 0 for meaningful stats.

---

## Important implementation notes

- **CIFS truncation + null bytes:** The Linux/Windows sandbox boundary corrupts large file
  writes. Two failure modes:
  1. File truncated mid-line (content cut off) — most common after Edit tool calls.
  2. Null bytes (`\x00`) appended after real content.
  After ANY write to train.py or debug_watch.py, ALWAYS run both:
  ```bash
  python3 -c "
  with open('FILE', 'rb') as f: c = f.read()
  c = c[:c.find(b'\x00')] if b'\x00' in c else c
  with open('FILE', 'wb') as f: f.write(c)
  " && python3 -c "import ast; ast.parse(open('FILE').read()); print('OK')"
  ```
  For appending missing tail content: use `cat >> file << 'EOF' ... EOF` heredoc.
  For full rewrites: use `cat > file << 'EOF' ... EOF` heredoc.
  If heredoc appends without a leading newline (merge on same line), fix with python3:
  ```bash
  python3 -c "
  with open('FILE') as f: c = f.read()
  c = c.replace('old_merged_line', 'fixed\nnewline')
  with open('FILE','w') as f: f.write(c)
  "
  ```
  Always use single-occurrence targeted edits — `replace_all=True` is more likely to corrupt.

- **Checkpoint validity:** Any checkpoint trained before the game.py owner-conversion fix
  is invalid. Do not resume from runs before that fix.

- **Zero-sum Bellman sign:** `target = r + γ(1-done)(−max Q̂(s'))` — the negative sign is
  intentional. `s'` is always encoded from the opponent's perspective.

- **Cascade simulation:** `_simulate_enemy_fire` and `_move_risk_score` (both in
  `rl/game_analysis.py`) use `game._snapshot()` / `game._restore()` to run headless
  cascades without mutating live state. Never call these without restoring. All three of
  train.py, debug_watch.py, and arena.py import from game_analysis — do not duplicate
  these functions elsewhere.

- **Self-play removed from curriculum:** Stage 4 is pure greedy (not self-play). After
  Stage 3 graduation the loop continues vs greedy. `run_episode_selfplay` still exists in
  train.py but is not called by default.

- **attack_chain_score encoding:** Stored in snapshot as
  `sum(REWARD_ATTACK_CHAIN / 2 for each directed attacking chain pair)`.
  The cap in shaping is `min(attack_chain_score, REWARD_ATTACK_CHAIN * CHAIN_REWARD_CAP)`.

- **phase_boost divisor is /10.0 (not /4.0 or /6.0):** `phase_boost = max(1.0, 3.0 - total_orbs / 10.0)`.
  Fades to 1× at ~20 total orbs. debug_watch.py matches this. Do not revert to /4.0 or /6.0.

---

## Known issues / design decisions

### Resolved
- **Q-value divergence:** `target_update=300` caused monotonically exploding loss. Fixed: 1000+.
- **Stage 1 too fast:** eps_decay_1=80k graduated the model before it had seen enough patterns.
  Fixed: eps_decay_1=150k.
- **Blocked action Q-values:** risky actions got no gradient without the penalty push.
  Fixed: `_exec_safe_action` pushes a penalty transition to buffer for each blocked action.
- **Unbounded chain rewards:** "full current score" mechanic with no cap caused Q-value inflation
  in Stage 2. Fixed: `CHAIN_REWARD_CAP = 5`.
- **sim_cascade overflow:** could reach −3.0+ per step, drowning all positive signals.
  Fixed: `SIM_CASCADE_MAX_PENALTY = 0.45` hard cap.
- **l_build misfiring:** fired from cascade-created accidental primes, not intentional setup;
  also re-fired each time a cascade rebuilt the same configuration. Removed entirely.
- **corner_setup vs corner_execute:** rewarding the setup (corner primed + adjacent owned)
  was wrong — that config is dangerous (exposure). Now we reward execution: corner fires
  post-cascade. `REWARD_CORNER_EXECUTE` replaces `REWARD_CORNER_SETUP`.
- **Stage 3 graduation gate too late:** GRAD_MIN_EPS[3]=0.30 opened the gate inside the
  crash zone. Fixed: raised to 0.45 — gate opens earlier, during post-crash recovery.
- **Stage 4 instability:** two crashes destroyed Stage 3 gains. Fixed: pure greedy opponent
  (removes mixed-policy gradient interference), STAGE_TARGET_UPDATE[4]=2000, STAGE_LR[4]=1e-5.
- **parse_args missing --resume:** train.py exited silently with AttributeError on args.resume.
  Fixed: added `--resume` argument to parse_args.
- **PENALTY_UNRESOLVED_EXPOSURE (Type 2):** flat per-chain-size penalty for ignoring
  exposure each turn created noisy gradients and conflicted with the risk filter.
  Removed entirely. Type 1 simulation + risk filter handle the same concern more precisely.
- **_move_risk_score summing all threats independently:** old logic over-counted danger by
  simulating each enemy primed cell firing separately and summing. Replaced with a single
  greedy-reply simulation (one enemy move) and a relative gain/loss comparison, active
  only when total_orbs >= 28. Block condition changed from `risk >= 4` to `risk > 0`.
- **Model not building chains / primes:** addressed with PENALTY_UNPRIMED_CELLS=0.030
  (fires midgame when owned_unprimed > 5), REWARD_SAFE_PRIME=0.050 (phase-boosted early
  game), and SIM_CASCADE_MAX_PENALTY cap. Confirmed resolved: final run achieves 95% vs
  random, 81.5% vs defensive, 57% vs greedy. Model definitively outperforms all prior
  checkpoints in arena (ep10000 final.pt: 55–68% win rate vs all earlier checkpoints).
- **Model not learning corner/edge strategy:** addressed with phase_boost on
  REWARD_CORNER_CONTROL and REWARD_CORNER_EXECUTE (3× early game). Resolved by final run.

### Active
- **Stage 2 crash with sliding-window buffer:** loss spikes in Stage 2 when S1 data
  (random opponent, no exposure punished) is still in the buffer while the defensive
  opponent starts punishing exposure. Mitigated by the final run's curriculum settings;
  no structural fix yet. If re-running, watch for loss > 0.8 in Stage 2 as a warning sign.

---

## TODO

Training is complete (`final.pt` — ep10000, 95% vs random, 81.5% vs defensive, 57% vs
greedy). Remaining work is game-side and cleanup only.

- **[x] Cleanup dead code in rl/ files** — remove `run_episode_selfplay` (never called),
  any stale imports, and other cruft left from earlier iterations. Python files only; do
  not delete trainold2.py / trainold3.py (kept as historical reference).
- **[x] Cleanup checkpoints folder** *(Ved)* — delete intermediate checkpoints that are
  now superseded. Keep: `final.pt` and a small set of milestone checkpoints for arena
  comparison. Everything else can go.
- **[ ] Add opponent selection to the game** — extend `main.py` so the player-vs-player
  menu offers: Human, Model (final.pt), Greedy (greedy_capture_policy), Defensive
  (defensive_policy). No more model training from the game UI — this is purely play mode.
- **[ ] Rebuild README.md and create README2.md** — `README.md` covers the program: how
  to install, run, play, and use the game. `README2.md` covers the model: training
  architecture, curriculum design, reward shaping history, known issues, and arena results.
  README.md should link to README2.md for the technical deep-dive.
