"""
arena.py  —  Headless model-vs-model evaluation.

Runs N games between two checkpoints, alternating seats, and reports
win/draw statistics with optional force_greedy and filter_moves logic
mirroring the training loop exactly.

Usage
-----
    python rl/arena.py --ckpt0 rl/checkpoints/ep05000.pt \
                       --ckpt1 rl/checkpoints/final.pt   \
                       --games 400                        \
                       --force_greedy --filter_moves

    python rl/arena.py --ckpt0 rl/checkpoints/ep06000.pt \
                       --ckpt1 rl/checkpoints/ep07000.pt

Seat alternation
----------------
  Even games : ckpt0 = Player 0, ckpt1 = Player 1
  Odd  games : ckpt0 = Player 1, ckpt1 = Player 0

Output
------
  Per-game winner line + summary table:

    Game   1  ckpt0 (P0) wins   [30 plies]
    Game   2  ckpt1 (P0) wins   [28 plies]
    ...
    ================================================
    Results after 400 games (200 as P0, 200 as P1)
    ------------------------------------------------
               Wins  As-P0  As-P1  Win%
    ckpt0       218    112    106  54.5%
    ckpt1       182     88     94  45.5%
    ================================================
"""

import argparse
import os
import sys

import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game
from rl.env           import ChainReactionEnv
from rl.model         import DQN
from rl.policies      import greedy_capture_policy
from rl.game_analysis import count_exposure, _has_big_cluster, _move_risk_score


def _select_action(model, env, player, state, mask, device,
                   force_greedy=False, filter_moves=False, epsilon=0.0):
    """
    Select an action for the given model.

    force_greedy : if True, override with greedy_capture_policy when a big
                   mixed primed cluster exists and the player has exposure.
    filter_moves : if True, block moves where our loss exceeds our gain
                   (relative risk filter); fall back to least-risky.
    epsilon      : probability of picking a random legal move (breaks determinism
                   so repeated games from identical positions diverge).
    """
    game = env.game

    # Force-greedy override (mirrors training loop) — takes priority over epsilon
    if force_greedy:
        exposure = count_exposure(game, player)
        if exposure > 0 and _has_big_cluster(game, player):
            r_act, c_act = greedy_capture_policy(game, player)
            return r_act * env.cols + c_act

    # Epsilon-random
    if epsilon > 0.0 and random.random() < epsilon:
        legal = np.where(mask)[0]
        return int(random.choice(legal))

    # Get Q-values
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_vals = model(state_t).squeeze(0).cpu().numpy()

    q_masked = q_vals.copy()
    q_masked[mask == 0] = -1e9

    if not filter_moves:
        return int(np.argmax(q_masked))

    # Risk filter: try actions from best to worst Q-value
    sorted_actions = np.argsort(q_masked)[::-1].tolist()
    legal_actions  = [a for a in sorted_actions if mask[a]]

    risk_scores = {}
    for action in legal_actions:
        r_a, c_a = divmod(action, env.cols)
        risk = _move_risk_score(game, player, r_a, c_a)
        if risk == 0.0:
            return action
        risk_scores[action] = risk

    # All moves risky — pick least damaging
    return min(risk_scores, key=risk_scores.get)


# ======================================================================
# Game runner
# ======================================================================

def run_game(model0, model1, env, device, args, game_idx):
    """
    Play one complete game.

    model0 is always 'ckpt0'; model1 is always 'ckpt1'.
    Seat assignment alternates by game_idx:
      even → ckpt0=P0, ckpt1=P1
      odd  → ckpt0=P1, ckpt1=P0

    Returns: (winner_label, seat_of_ckpt0, plies)
      winner_label : 'ckpt0' | 'ckpt1' | 'draw'
      seat_of_ckpt0: 0 or 1
    """
    seat0 = game_idx % 2           # seat that ckpt0 occupies
    seat1 = 1 - seat0              # seat that ckpt1 occupies
    models = {seat0: model0, seat1: model1}

    state, _ = env.reset()
    done      = False
    plies     = 0
    guard     = env.rows * env.cols * 10

    while not done and plies < guard:
        player = env.game.current_player
        mask   = env.valid_action_mask(player)
        model  = models[player]

        action = _select_action(
            model, env, player, state, mask, device,
            force_greedy=args.force_greedy,
            filter_moves=args.filter_moves,
            epsilon=args.epsilon,
        )

        state, _, done, _, _ = env.step(action)
        plies += 1

    winner_seat = env.game.winner
    if winner_seat is None:
        return 'draw', seat0, plies
    if winner_seat == seat0:
        return 'ckpt0', seat0, plies
    else:
        return 'ckpt1', seat0, plies


# ======================================================================
# Main
# ======================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    def load_model(path, rows, cols, channels):
        net = DQN(rows, cols, channels).to(device)
        ckpt = torch.load(path, map_location=device)
        # Support both raw state_dict and wrapped checkpoint formats
        if isinstance(ckpt, dict) and 'q_net' in ckpt:
            net.load_state_dict(ckpt['q_net'])
        elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        return net

    print(f"\nArena: {os.path.basename(args.ckpt0)}  vs  {os.path.basename(args.ckpt1)}")
    print(f"Games: {args.games}  |  epsilon={args.epsilon}  force_greedy={args.force_greedy}  filter_moves={args.filter_moves}")
    print(f"Board: {args.rows}x{args.cols}  channels={args.channels}  device={device}\n")

    model0 = load_model(args.ckpt0, args.rows, args.cols, args.channels)
    model1 = load_model(args.ckpt1, args.rows, args.cols, args.channels)

    env = ChainReactionEnv(rows=args.rows, cols=args.cols, num_players=2)

    # Stats
    wins   = {'ckpt0': 0, 'ckpt1': 0, 'draw': 0}
    as_p0  = {'ckpt0': 0, 'ckpt1': 0}   # wins when playing as P0
    as_p1  = {'ckpt0': 0, 'ckpt1': 0}   # wins when playing as P1

    label_w = max(len(os.path.basename(args.ckpt0)),
                  len(os.path.basename(args.ckpt1)), 6)

    if args.verbose:
        print(f"{'Game':>6}  {'Winner':<{label_w+4}}  {'Plies':>5}")
        print("-" * (6 + label_w + 4 + 7 + 4))

    for i in range(args.games):
        winner, seat0, plies = run_game(model0, model1, env, device, args, i)
        wins[winner] += 1

        ckpt0_seat = seat0   # seat ckpt0 is playing as in this game
        ckpt1_seat = 1 - seat0

        if winner == 'ckpt0':
            if ckpt0_seat == 0:
                as_p0['ckpt0'] += 1
            else:
                as_p1['ckpt0'] += 1
        elif winner == 'ckpt1':
            if ckpt1_seat == 0:
                as_p0['ckpt1'] += 1
            else:
                as_p1['ckpt1'] += 1

        if args.verbose:
            seat_str = f"(P{ckpt0_seat})" if winner == 'ckpt0' else f"(P{ckpt1_seat})"
            win_str  = f"{winner} {seat_str} wins" if winner != 'draw' else "draw"
            print(f"{i+1:>6}  {win_str:<{label_w+4}}  {plies:>5} plies")

    # Summary
    n0    = os.path.basename(args.ckpt0)
    n1    = os.path.basename(args.ckpt1)
    nw    = max(len(n0), len(n1), 8)
    total = args.games

    print(f"\n{'='*(nw + 40)}")
    print(f"Results after {total} games  "
          f"({total//2} as P0, {total//2} as P1 each)")
    print(f"{'-'*(nw + 40)}")
    print(f"{'Model':<{nw}}  {'Wins':>5}  {'As-P0':>6}  {'As-P1':>6}  {'Win%':>6}")
    print(f"{'-'*(nw + 40)}")
    for label, name in [('ckpt0', n0), ('ckpt1', n1)]:
        w   = wins[label]
        p0  = as_p0[label]
        p1  = as_p1[label]
        pct = 100 * w / total if total else 0
        print(f"{name:<{nw}}  {w:>5}  {p0:>6}  {p1:>6}  {pct:>5.1f}%")
    if wins['draw']:
        print(f"{'draws':<{nw}}  {wins['draw']:>5}")
    print(f"{'='*(nw + 40)}\n")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Headless model-vs-model arena evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt0",         type=str,  required=True,
                   help="Path to first checkpoint (ckpt0)")
    p.add_argument("--ckpt1",         type=str,  required=True,
                   help="Path to second checkpoint (ckpt1)")
    p.add_argument("--games",         type=int,  default=400,
                   help="Total games to play (should be even for fair seat split)")
    p.add_argument("--rows",          type=int,  default=6)
    p.add_argument("--cols",          type=int,  default=6)
    p.add_argument("--channels",      type=int,  default=64)
    p.add_argument("--epsilon",        type=float, default=0.05,
                                   help="Random-move probability for both models (breaks determinism)")
    p.add_argument("--force_greedy",  action="store_true",
                   help="Mirror training force-greedy override (fire when big cluster + exposure)")
    p.add_argument("--filter_moves",  action="store_true",
                   help="Mirror training risk filter (block net-negative trades)")
    p.add_argument("--verbose",       action="store_true",
                   help="Print per-game results")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
