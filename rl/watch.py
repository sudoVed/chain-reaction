"""
watch.py  -  Watch trained DQN agents play Chain Reaction visually.

Usage
-----
    # AI checkpoint vs greedy policy
    python rl/watch.py --ckpt0 rl/checkpoints/final.pt --policy1 greedy

    # Two AI checkpoints head-to-head
    python rl/watch.py --ckpt0 rl/checkpoints/ep02000.pt --ckpt1 rl/checkpoints/final.pt

    # Human vs AI
    python rl/watch.py --ckpt0 rl/checkpoints/final.pt --human 1

    # AI vs defensive policy, slower moves
    python rl/watch.py --ckpt0 rl/checkpoints/final.pt --policy1 defensive --delay 800

Controls
--------
    Space / Enter  : pause / unpause
    R              : restart
    Esc            : quit
"""

import argparse
import os
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game      import Game
from renderer  import GameRenderer
from constants import FPS
from rl.env      import ChainReactionEnv
from rl.agent    import DQNAgent
from rl.policies import random_policy, greedy_capture_policy, defensive_policy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

NAMED_POLICIES = {
    "random":    random_policy,
    "greedy":    greedy_capture_policy,
    "defensive": defensive_policy,
}

# ---------------------------------------------------------------------------
# BGM playlist
# ---------------------------------------------------------------------------

_BGM_TRACKS = [
    os.path.join(ASSETS_DIR, "bgm.mp3"),
    os.path.join(ASSETS_DIR, "bgm2.mp3"),
]
_BGM_INDEX = 0
BGM_END    = pygame.USEREVENT + 1


def _start_bgm():
    global _BGM_INDEX
    _BGM_INDEX = 0
    _play_bgm_track(_BGM_INDEX)


def _play_bgm_track(index: int):
    try:
        pygame.mixer.music.load(_BGM_TRACKS[index % len(_BGM_TRACKS)])
        pygame.mixer.music.set_volume(0.35)
        pygame.mixer.music.set_endevent(BGM_END)
        pygame.mixer.music.play(0)
    except Exception:
        pass


def _advance_bgm():
    global _BGM_INDEX
    _BGM_INDEX = (_BGM_INDEX + 1) % len(_BGM_TRACKS)
    _play_bgm_track(_BGM_INDEX)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_agent(path: str, env: ChainReactionEnv) -> DQNAgent:
    """
    Load a DQNAgent from a checkpoint.
    Falls back to a freshly-initialised random agent if path is None.
    """
    if path and os.path.exists(path):
        print(f"  Loading agent from {path}")
        return DQNAgent.from_checkpoint(path)
    else:
        print(f"  No checkpoint at {path!r} — using random agent")
        return DQNAgent(rows=env.rows, cols=env.cols)


def ai_pick_action(agent: DQNAgent, env: ChainReactionEnv,
                   player: int, epsilon: float = 0.0) -> int:
    """Ask the agent for an action index for the given player."""
    state = env.encode_state(player)
    mask  = env.valid_action_mask(player)
    return agent.select_action(state, mask, epsilon)


# ------------------------------------------------------------------
# Main watch loop
# ------------------------------------------------------------------

def watch(args: argparse.Namespace):
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    pygame.display.set_caption("Chain Reaction — Watch")

    icon_path = os.path.join(ASSETS_DIR, "icon.png")
    if os.path.exists(icon_path):
        try:
            pygame.display.set_icon(pygame.image.load(icon_path))
        except Exception:
            pass

    screen = pygame.display.set_mode((860, 760), pygame.RESIZABLE)
    clock  = pygame.time.Clock()

    _start_bgm()

    human_turns = set(args.human)
    ckpt0 = args.ckpt or args.ckpt0
    ckpt1 = args.ckpt or args.ckpt1

    # Read board size from checkpoint if available
    any_ckpt = ckpt0 or ckpt1
    if any_ckpt and os.path.exists(any_ckpt):
        import torch
        meta = torch.load(any_ckpt, map_location="cpu", weights_only=False)
        rows = meta.get("rows", args.rows)
        cols = meta.get("cols", args.cols)
        print(f"  Board size from checkpoint: {rows}x{cols}")
    else:
        rows, cols = args.rows, args.cols

    env = ChainReactionEnv(rows=rows, cols=cols, num_players=2)

    # Build agent list — None for human seats
    agent0 = load_agent(ckpt0, env) if 0 not in human_turns else None
    agent1 = load_agent(ckpt1, env) if 1 not in human_turns else None
    agents = [agent0, agent1]

    # Build policy list — overrides agent for that seat if set
    policies = [
        NAMED_POLICIES.get(args.policy0),
        NAMED_POLICIES.get(args.policy1),
    ]

    # Describe what's playing
    def seat_label(i):
        if i in human_turns:    return "Human"
        if policies[i]:         return f"Policy({args.policy0 if i==0 else args.policy1})"
        ckpt = ckpt0 if i == 0 else ckpt1
        return os.path.basename(ckpt) if ckpt else "RandomNet"

    print(f"\n  P0: {seat_label(0)}  vs  P1: {seat_label(1)}")
    print(f"  Delay={args.delay}ms  Space=pause  R=restart  Esc=quit\n")

    paused = False

    def new_game():
        game     = Game(2, rows, cols)
        renderer = GameRenderer(screen, game)
        env.game = game
        return game, renderer

    game, renderer = new_game()
    ai_timer = 0
    wins = [0, 0]

    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == BGM_END:
                _advance_bgm()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    paused = not paused
                    print("Paused." if paused else "Resumed.")
                if event.key == pygame.K_r:
                    game, renderer = new_game()
                    ai_timer = 0

            if game.state == "placing" and game.current_player in human_turns:
                renderer.handle_event(event)

        if paused:
            renderer.draw()
            continue

        # AI / policy move
        if game.state == "placing" and game.current_player not in human_turns:
            ai_timer += dt
            if ai_timer >= args.delay:
                ai_timer = 0
                player    = game.current_player
                policy_fn = policies[player]

                if policy_fn is not None:
                    r, c = policy_fn(game, player)
                else:
                    action = ai_pick_action(agents[player], env, player, args.epsilon)
                    r, c   = divmod(action, game.cols)

                if game.can_place(r, c):
                    game.place(r, c)
                    renderer._phase_timer = 0

        # Auto-restart after win
        if game.state == "won":
            renderer.draw()
            pygame.time.wait(args.restart_delay)
            w = game.winner
            wins[w] += 1
            print(f"  {seat_label(w)} wins!  Score: {seat_label(0)} {wins[0]} – {wins[1]} {seat_label(1)}")
            game, renderer = new_game()
            ai_timer = 0
            continue

        renderer.update(dt)
        renderer.draw()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch Chain Reaction agents play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt",      type=str, default=None,
                   help="Single checkpoint for both seats")
    p.add_argument("--ckpt0",     type=str, default=None,
                   help="Checkpoint for Player 0")
    p.add_argument("--ckpt1",     type=str, default=None,
                   help="Checkpoint for Player 1")
    p.add_argument("--policy0",   type=str, default=None,
                   choices=["random", "greedy", "defensive"],
                   help="Fixed policy for Player 0 (overrides ckpt0)")
    p.add_argument("--policy1",   type=str, default=None,
                   choices=["random", "greedy", "defensive"],
                   help="Fixed policy for Player 1 (overrides ckpt1)")
    p.add_argument("--rows",      type=int, default=6)
    p.add_argument("--cols",      type=int, default=6)
    p.add_argument("--human",     type=int, nargs="*", default=[],
                   help="Seat indices for human players (e.g. --human 1)")
    p.add_argument("--delay",     type=int, default=600,
                   help="ms between AI moves")
    p.add_argument("--restart_delay", type=int, default=2000)
    p.add_argument("--epsilon",   type=float, default=0.0,
                   help="Exploration noise for network players (0=greedy)")
    return p.parse_args()


if __name__ == "__main__":
    watch(parse_args())
