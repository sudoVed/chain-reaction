"""
env.py  -  Gym-style Chain Reaction environment for RL training.

State encoding  (always from the CURRENT player's perspective)
-------------------------------------------------------------
Flat float32 array of length 4 * rows * cols, four channels stacked:
  [0      ..   R*C)   my_count[r,c]     / 4.0   (0 if cell not mine)
  [R*C    .. 2*R*C)   enemy_count[r,c]  / 4.0   (0 if not enemy's)
  [2*R*C  .. 3*R*C)   critical_mass[r,c]/ 4.0   (static geometry)
  [3*R*C  .. 4*R*C)   primed_map[r,c]            +1.0 if MY primed cell
                                                  -1.0 if ENEMY primed cell
                                                   0.0 otherwise

The primed map lets the CNN directly see +1/-1 adjacencies:
  +1 next to -1  =>  exposure (danger/opportunity)
  +1 next to +1  =>  your chain building
  -1 next to -1  =>  enemy chain nearby

Action space
------------
Integer in [0, rows*cols).  action = r*cols + c.
Invalid actions (cells owned by opponent) are masked to -inf in the
agent's Q-values before argmax, so they are never selected.

Terminal reward scheme
----------------------
  +5.0   if the player who just moved wins (cascade resolved to win)
   0.0   all other steps
  -1.0   penalty for submitting an invalid action (should not occur)

The self-play training loop negates the reward when storing the
opponent's perspective  (zero-sum: my +3 = opponent's -3).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game import Game


class ChainReactionEnv:
    """
    Wraps game.Game as a two-player self-play RL environment.

    Both players share the same state encoding convention: the board is
    always described from the perspective of whoever is about to move.
    This lets a single neural network play both sides without any
    architecture changes.
    """

    def __init__(self, rows: int = 9, cols: int = 9, num_players: int = 2):
        self.rows        = rows
        self.cols        = cols
        self.num_players = num_players
        self.action_size = rows * cols
        self.state_size  = 4 * rows * cols   # 4 channels now

        self.game: Game | None = None
        self._cm_channel = self._precompute_cm()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _precompute_cm(self) -> np.ndarray:
        """
        Build the critical-mass channel once at init time.

        Returns float32 array of length rows*cols where each entry is
        critical_mass(r,c) / 4.0.  Corner=0.5, edge=0.75, interior=1.0.
        """
        tmp = Game(2, self.rows, self.cols)
        cm  = np.zeros(self.rows * self.cols, dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                cm[r * self.cols + c] = tmp.critical_mass(r, c) / 4.0
        return cm

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        """Start a fresh game. Returns (state, info)."""
        self.game = Game(self.num_players, self.rows, self.cols)
        return self.encode_state(self.game.current_player), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one placement for the current player.

        Simulates the full cascade before returning.  next_state is
        encoded from the NEW current player's perspective.

        Returns:
            next_state, reward, done, truncated, info
        """
        g              = self.game
        current_player = g.current_player
        r, c           = divmod(action, self.cols)

        if not g.can_place(r, c):
            next_state = self.encode_state(current_player)
            return next_state, -1.0, True, False, {"winner": -1, "invalid": True}

        g.place(r, c)
        self._simulate_cascade()

        done   = (g.state == "won")
        reward = 5.0 if (done and g.winner == current_player) else 0.0

        next_player = g.current_player
        next_state  = self.encode_state(next_player)

        return next_state, reward, done, False, {
            "winner":      g.winner if done else -1,
            "next_player": next_player,
        }

    # ------------------------------------------------------------------
    # State / action helpers
    # ------------------------------------------------------------------

    def encode_state(self, player: int) -> np.ndarray:
        """
        Return the board as a normalised float32 vector from `player`'s view.

        Channel 0: my orb counts    / 4  (0 where not mine)
        Channel 1: enemy orb counts / 4  (0 where not enemy's)
        Channel 2: critical mass    / 4  (static geometry)
        Channel 3: primed map — +1.0 my primed, -1.0 enemy primed, 0.0 else
        """
        g            = self.game
        my_counts    = np.zeros(self.rows * self.cols, dtype=np.float32)
        enemy_counts = np.zeros(self.rows * self.cols, dtype=np.float32)
        primed_map   = np.zeros(self.rows * self.cols, dtype=np.float32)

        for r in range(self.rows):
            for c in range(self.cols):
                cell = g.grid[r][c]
                idx  = r * self.cols + c
                if cell.owner == player:
                    my_counts[idx] = cell.count / 4.0
                    if g.is_primed(r, c):
                        primed_map[idx] = 1.0
                elif cell.owner >= 0:
                    enemy_counts[idx] = cell.count / 4.0
                    if g.is_primed(r, c):
                        primed_map[idx] = -1.0

        return np.concatenate([my_counts, enemy_counts,
                                self._cm_channel, primed_map])

    def valid_action_mask(self, player: int) -> np.ndarray:
        """
        Boolean array of length action_size.
        True = legal placement for `player`.
        """
        g    = self.game
        mask = np.zeros(self.action_size, dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = g.grid[r][c]
                if cell.is_empty() or cell.owner == player:
                    mask[r * self.cols + c] = True
        return mask

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simulate_cascade(self):
        """Drive every explosion wave to completion after a placement."""
        safety = 0
        while self.game.state == "animating":
            wave = self.game.get_wave()
            if not wave:
                break
            self.game.apply_wave(wave)
            safety += 1
            if safety > 10_000:
                raise RuntimeError("Infinite cascade detected in env!")
