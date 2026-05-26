"""
policies.py  -  Fixed (non-learning) opponent policies for curriculum training.

Each policy is a callable:
    policy(game: Game, player: int) -> (row: int, col: int)

Policies are stateless and deterministic given a random seed.  They are
used as training opponents in the curriculum phases before self-play, and
can also be passed to watch.py for human vs fixed-policy games.

Curriculum ladder (weakest → strongest):
    1. random_policy          — uniform random legal move
    2. greedy_capture_policy  — prefer own primed cells; bonus for adjacent
                                enemy primed (triggers their chain too)
    3. defensive_policy       — avoid handing the opponent free cascades;
                                among safe moves, prefer own primed cells
"""

import random
from game import Game


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _legal_moves(game: Game, player: int) -> list:
    """Return all (r, c) the player may legally place on."""
    return [
        (r, c)
        for r in range(game.rows)
        for c in range(game.cols)
        if game.can_place(r, c)
    ]


def _enemy_primed_set(game: Game, player: int) -> set:
    """Return the set of (r, c) where the opponent has a primed cell."""
    enemy = 1 - player
    return {
        (r, c)
        for r in range(game.rows)
        for c in range(game.cols)
        if game.grid[r][c].owner == enemy and game.is_primed(r, c)
    }


# ------------------------------------------------------------------
# Policy 1: random
# ------------------------------------------------------------------

def random_policy(game: Game, player: int) -> tuple:
    """
    Pick any legal cell uniformly at random.

    Baseline opponent — used in Stage 1 of the curriculum.  Provides
    simple, dense feedback: the agent just needs to beat chaos.
    """
    moves = _legal_moves(game, player)
    return random.choice(moves)


# ------------------------------------------------------------------
# Policy 2: greedy capture
# ------------------------------------------------------------------

def greedy_capture_policy(game: Game, player: int) -> tuple:
    """
    Prefer placing on own primed cells; among those prefer ones adjacent
    to enemy primed cells so they trigger the enemy's chain too.

    Priority order:
        1. My primed cell adjacent to ≥1 enemy primed cell  (attack + steal)
        2. Any of my primed cells                           (build chains)
        3. Any legal move                                   (fallback)

    Used in Stage 2 of the curriculum.  Teaches the agent to defend
    against an opponent that actively tries to trigger chains.
    """
    moves      = _legal_moves(game, player)
    enemy_pset = _enemy_primed_set(game, player)

    # Tier 1: own primed cells that border an enemy primed cell
    tier1 = [
        (r, c) for r, c in moves
        if game.is_primed(r, c)
        and any((nr, nc) in enemy_pset for nr, nc in game.neighbours(r, c))
    ]
    if tier1:
        return random.choice(tier1)

    # Tier 2: any own primed cell
    tier2 = [(r, c) for r, c in moves if game.is_primed(r, c)]
    if tier2:
        return random.choice(tier2)

    # Tier 3: fallback to random
    return random.choice(moves)


# ------------------------------------------------------------------
# Policy 3: defensive
# ------------------------------------------------------------------

def defensive_policy(game: Game, player: int) -> tuple:
    """
    Avoid placing adjacent to enemy primed cells (would hand them a
    free cascade into our territory).  Among safe moves, prefer own
    primed cells first (greedy_capture logic), then anything else.

    Priority order:
        1. Safe own primed cell adjacent to enemy primed  (preemptive blast)
        2. Any safe own primed cell                       (safe chain-building)
        3. Any safe move                                  (neutral but safe)
        4. Dangerous own primed cell                      (last resort attack)
        5. Any legal move                                 (absolute fallback)

    Used in Stage 3 of the curriculum.  Teaches the agent to recognise
    exposure (own primed adjacent to enemy primed) and handle it without
    being forced into a losing trade.
    """
    moves      = _legal_moves(game, player)
    enemy_pset = _enemy_primed_set(game, player)

    def is_dangerous(r, c):
        """True if placing here puts us adjacent to an enemy primed cell."""
        return any((nr, nc) in enemy_pset for nr, nc in game.neighbours(r, c))

    safe_moves = [(r, c) for r, c in moves if not is_dangerous(r, c)]

    # Tier 1: safe own primed cell that also borders enemy primed (preempt)
    tier1 = [
        (r, c) for r, c in safe_moves
        if game.is_primed(r, c)
        and any((nr, nc) in enemy_pset for nr, nc in game.neighbours(r, c))
    ]
    if tier1:
        return random.choice(tier1)

    # Tier 2: safe own primed cell
    tier2 = [(r, c) for r, c in safe_moves if game.is_primed(r, c)]
    if tier2:
        return random.choice(tier2)

    # Tier 3: any safe move
    if safe_moves:
        return random.choice(safe_moves)

    # Tier 4: dangerous own primed (no safe option, at least attack)
    tier4 = [(r, c) for r, c in moves if game.is_primed(r, c)]
    if tier4:
        return random.choice(tier4)

    # Tier 5: absolute fallback
    return random.choice(moves)
