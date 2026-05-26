"""
game_analysis.py  —  Pure board analysis helpers.

Shared across train.py, arena.py, debug_watch.py and any other module
that needs to inspect or simulate game state without caring about training.

No reward constants, no replay buffer, no curriculum logic here.
All functions are stateless and use game._snapshot() / game._restore()
to avoid mutating live state.

Functions
---------
  _primed_components(game)              -> list[set[(r,c)]]
  count_exposure(game, player)          -> int
  count_primed(game, player)            -> int
  _has_big_cluster(game, player)        -> bool
  _simulate_enemy_fire(game, er, ec, p) -> int   (orb loss)
  _move_risk_score(game, player, r, c)  -> float (block if > 0)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.policies import greedy_capture_policy


# ------------------------------------------------------------------
# Connected-component helpers
# ------------------------------------------------------------------

def _primed_components(game):
    """
    Connected components of ALL primed cells across both players (4-connectivity).
    Returns a list of sets of (r, c).
    Used for greedy override condition and chain-size scaling.
    """
    all_primed = {
        (r, c)
        for r in range(game.rows) for c in range(game.cols)
        if game.grid[r][c].owner is not None and game.is_primed(r, c)
    }
    visited    = set()
    components = []
    for start in all_primed:
        if start in visited:
            continue
        comp     = set()
        frontier = [start]
        while frontier:
            pos = frontier.pop()
            if pos in visited:
                continue
            visited.add(pos)
            comp.add(pos)
            for nb in game.neighbours(*pos):
                if nb in all_primed and nb not in visited:
                    frontier.append(nb)
        components.append(comp)
    return components


# ------------------------------------------------------------------
# Exposure helpers
# ------------------------------------------------------------------

def count_exposure(game, player):
    """Count player's primed cells that are adjacent to at least one enemy primed cell."""
    enemy        = 1 - player
    enemy_primed = {(r, c) for r in range(game.rows) for c in range(game.cols)
                    if game.grid[r][c].owner == enemy and game.is_primed(r, c)}
    return sum(
        1 for r in range(game.rows) for c in range(game.cols)
        if game.grid[r][c].owner == player and game.is_primed(r, c)
        and any(nb in enemy_primed for nb in game.neighbours(r, c))
    )


def count_primed(game, player):
    """Count all primed cells owned by player."""
    return sum(
        1 for r in range(game.rows) for c in range(game.cols)
        if game.grid[r][c].owner == player and game.is_primed(r, c)
    )


def _has_big_cluster(game, player):
    """
    True if any mixed primed component has >= 4 cells AND contains >= 1 cell
    owned by player.  Triggers the forced greedy fire override in all stages.
    """
    my_primed = {(r, c) for r in range(game.rows) for c in range(game.cols)
                 if game.grid[r][c].owner == player and game.is_primed(r, c)}
    return any(
        len(comp) >= 4 and any(cell in my_primed for cell in comp)
        for comp in _primed_components(game)
    )


# ------------------------------------------------------------------
# Cascade simulation helpers
# ------------------------------------------------------------------

def _simulate_enemy_fire(game, er, ec, agent_player):
    """
    Snapshot, push enemy primed cell at (er, ec) over critical mass, run cascade,
    measure agent orb loss, restore.  Called once per threatening enemy cell.
    Used by compute_shaping (Type 1 penalty) in train.py.
    """
    orbs_before = sum(cell.count for row in game.grid for cell in row
                      if cell.owner == agent_player)
    snap = game._snapshot()

    game.grid[er][ec].count += 1
    game.state = "animating"
    game.explosion_queue.append((er, ec))
    safety = 0
    while game.state == "animating" and safety < 10_000:
        wave = game.get_wave()
        if not wave:
            break
        game.apply_wave(wave)
        safety += 1

    orbs_after = sum(cell.count for row in game.grid for cell in row
                     if cell.owner == agent_player)
    game._restore(snap)
    return max(orbs_before - orbs_after, 0)


# ------------------------------------------------------------------
# Risk filter
# ------------------------------------------------------------------

def _move_risk_score(game, player, r, c, reply_fn=None):
    """
    Relative risk filter — active once total_orbs >= 20 (midgame).

    Three-snapshot evaluation:
      1. Before our move    — baseline primed/unprimed counts
      2. After our cascade  — our_gain = 1.0*primed_gained + 0.2*unprimed_gained
      3. After enemy reply  — our_loss = 1.0*primed_lost + 0.2*unprimed_lost

    reply_fn(game, enemy) -> (r, c): callable that picks the opponent's response.
    Defaults to greedy_capture_policy when None (used in training).
    Pass the model itself for self-aware lookahead in play mode.

    Returns max(0, our_loss - our_gain).
    Block the move if the return value > 0 (trade is net negative for us).
    Returns 0.0 unconditionally if total_orbs < 20 or our move wins the game.
    """
    if reply_fn is None:
        reply_fn = greedy_capture_policy

    if sum(game.orb_counts()) < 20:
        return 0.0

    enemy = 1 - player
    snap  = game._snapshot()

    # -- Snapshot 1: before our move --
    my_primed_start   = sum(1 for rr in range(game.rows) for cc in range(game.cols)
                            if game.grid[rr][cc].owner == player and game.is_primed(rr, cc))
    my_unprimed_start = sum(1 for rr in range(game.rows) for cc in range(game.cols)
                            if game.grid[rr][cc].owner == player and not game.is_primed(rr, cc))

    # -- Simulate our placement + cascade --
    game.grid[r][c].owner  = player
    game.grid[r][c].count += 1
    if game.grid[r][c].count >= game.critical_mass(r, c):
        game.state = "animating"
        game.explosion_queue.append((r, c))
        safety = 0
        while game.state == "animating" and safety < 10_000:
            wave = game.get_wave()
            if not wave:
                break
            game.apply_wave(wave)
            safety += 1

    # If our move already won the game there is no enemy reply
    if game.winner is not None:
        game._restore(snap)
        return 0.0

    # -- Snapshot 2: after our cascade --
    my_primed_mid   = sum(1 for rr in range(game.rows) for cc in range(game.cols)
                          if game.grid[rr][cc].owner == player and game.is_primed(rr, cc))
    my_unprimed_mid = sum(1 for rr in range(game.rows) for cc in range(game.cols)
                          if game.grid[rr][cc].owner == player and not game.is_primed(rr, cc))

    our_gain = (1.0 * max(my_primed_mid   - my_primed_start,   0)
              + 0.2 * max(my_unprimed_mid - my_unprimed_start, 0))

    # -- Enemy's single best retaliatory move --
    er, ec = reply_fn(game, enemy)
    game.grid[er][ec].count += 1
    if game.grid[er][ec].count >= game.critical_mass(er, ec):
        game.state = "animating"
        game.explosion_queue.append((er, ec))
        safety = 0
        while game.state == "animating" and safety < 10_000:
            wave = game.get_wave()
            if not wave:
                break
            game.apply_wave(wave)
            safety += 1

    # -- Snapshot 3: after enemy reply — measure our loss --
    my_primed_after   = sum(1 for rr in range(game.rows) for cc in range(game.cols)
                            if game.grid[rr][cc].owner == player and game.is_primed(rr, cc))
    my_unprimed_after = sum(1 for rr in range(game.rows) for cc in range(game.cols)
                            if game.grid[rr][cc].owner == player and not game.is_primed(rr, cc))

    our_loss = (1.0 * max(my_primed_mid   - my_primed_after,   0)
              + 0.2 * max(my_unprimed_mid - my_unprimed_after, 0))

    game._restore(snap)
    return max(0.0, our_loss - our_gain)
