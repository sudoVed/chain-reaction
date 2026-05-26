"""
ai_player.py  —  AI opponent for Chain Reaction (play mode).

Wraps the three fixed policies and the trained DQN model behind a single
pick_move(game, player) interface.

Modes
-----
  defensive : defensive_policy — avoids exposure, safe chain-building
  greedy    : greedy_capture_policy — fires primed cells aggressively
  smart     : DQN final.pt with force_greedy + filter_moves always on,
               mirroring the training overrides exactly

The smart agent always applies:
  - force_greedy : if exposure > 0 and a big mixed cluster exists, fire
                   immediately via greedy_capture_policy
  - filter_moves : block moves where our loss exceeds our gain (risk filter,
                   active from total_orbs >= 28)
"""

import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rl.policies      import greedy_capture_policy, defensive_policy
from rl.game_analysis import count_exposure, _has_big_cluster, _move_risk_score
from rl.env           import ChainReactionEnv
from rl.model         import DQN


# Path to the trained checkpoint
_CHECKPOINT = os.path.join(BASE_DIR, "rl", "final", "final.pt")


class AIPlayer:
    """
    AI opponent for human-vs-AI play.

    Args:
        mode  : "defensive" | "greedy" | "smart"
        rows  : board rows (passed through to env for smart mode)
        cols  : board cols
    """

    def __init__(self, mode: str, rows: int, cols: int):
        self.mode = mode
        self._model: DQN | None = None
        self._env:   ChainReactionEnv | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == "smart":
            self._env = ChainReactionEnv(rows=rows, cols=cols, num_players=2)
            ckpt = torch.load(_CHECKPOINT, map_location=self._device, weights_only=False)
            r = ckpt.get("rows", rows)
            c = ckpt.get("cols", cols)
            self._model = DQN(r, c).to(self._device)
            self._model.load_state_dict(ckpt["q_net"])
            self._model.eval()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pick_move(self, game, player: int) -> tuple[int, int]:
        """
        Return (row, col) for the AI to place an orb.

        Called by the game loop when it is the AI's turn.
        `game` is the live Game object; `player` is the AI's player index.
        """
        if self.mode == "defensive":
            return defensive_policy(game, player)
        if self.mode == "greedy":
            return greedy_capture_policy(game, player)
        return self._pick_smart(game, player)

    # ------------------------------------------------------------------
    # Smart (DQN) implementation
    # ------------------------------------------------------------------

    def _pick_smart(self, game, player: int) -> tuple[int, int]:
        """
        DQN move selection with force_greedy and filter_moves always on.

        1. Force-greedy override: if the player has exposure AND a large
           mixed primed cluster exists (>= 4 cells), fire immediately via
           greedy_capture_policy — same as training.
        2. Get Q-values from the network.
        3. Risk filter: iterate actions best→worst Q; return the first one
           whose _move_risk_score == 0 (safe trade). If all are risky, pick
           the least-damaging one.
        """
        # -- Force-greedy override --
        if count_exposure(game, player) > 0 and _has_big_cluster(game, player):
            return greedy_capture_policy(game, player)

        # -- Encode state --
        self._env.game = game
        state = self._env.encode_state(player)
        mask  = self._env.valid_action_mask(player)

        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self._model(state_t).squeeze(0).cpu().numpy()

        q_masked = q_vals.copy()
        q_masked[~mask] = -1e9

        # -- Self-aware reply function for risk filter --
        # Simulates what the model itself would play as the opponent,
        # giving a much better lookahead than a fixed greedy heuristic.
        def _model_reply(sim_game, enemy):
            self._env.game = sim_game
            s = self._env.encode_state(enemy)
            m = self._env.valid_action_mask(enemy)
            s_t = torch.tensor(s, dtype=torch.float32,
                               device=self._device).unsqueeze(0)
            with torch.no_grad():
                qv = self._model(s_t).squeeze(0).cpu().numpy()
            qv[~m] = -1e9
            return divmod(int(np.argmax(qv)), sim_game.cols)

        # -- Risk filter (model self-simulation) --
        sorted_actions = np.argsort(q_masked)[::-1].tolist()
        legal_actions  = [a for a in sorted_actions if mask[a]]

        risk_scores = {}
        for action in legal_actions:
            r_a, c_a = divmod(action, game.cols)
            risk = _move_risk_score(game, player, r_a, c_a, reply_fn=_model_reply)
            self._env.game = game   # restore after simulation
            if risk == 0.0:
                return r_a, c_a   # first safe action in Q-value order
            risk_scores[action] = risk

        # All moves carry risk — pick least damaging
        self._env.game = game
        best = min(risk_scores, key=risk_scores.get)
        return divmod(best, game.cols)
