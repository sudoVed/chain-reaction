"""
debug_watch.py  —  Visual step-by-step reward debugger.

Usage
-----
    python rl/debug_watch.py --ckpt0 rl/checkpoints/final.pt --policy1 greedy
    python rl/debug_watch.py --policy0 greedy --policy1 greedy

Controls
--------
    Space / Enter  : advance one move
    U              : undo last move
    R              : restart game
    Esc            : quit

Panel shows every reward/penalty component for the last move made,
plus a running cumulative total per player.

Flags
-----
    --force_greedy   Mirror training override: when exposure > 0 and a big mixed
                     cluster (>= 4 cells) exists, agent's move is replaced with
                     greedy_capture_policy (same as training).
    --filter_moves   Mirror training risk filter: block moves where our loss
                     exceeds our gain from the trade (midgame only, >= 28 orbs).
"""

import argparse
import copy
import os
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game      import Game
from renderer  import GameRenderer
from constants import FPS, PLAYER_COLORS
from rl.env      import ChainReactionEnv
from rl.agent    import DQNAgent
from rl.policies import random_policy, greedy_capture_policy, defensive_policy
from rl.game_analysis import (
    count_exposure, count_primed,
    _simulate_enemy_fire,
    _has_big_cluster,
    _move_risk_score,
)
from rl.train import (
    _positional_snapshot, wasted_placement_penalty,
    REWARD_CLEAN_CAPTURE, REWARD_CASCADE_OWN,
    REWARD_SAFE_PRIME, REWARD_ANY_PRIME,
    REWARD_CHAIN, REWARD_SAFE_ATTACK, REWARD_ATTACK_CHAIN,
    REWARD_CORNER_CONTROL, REWARD_CORNER_EXECUTE, REWARD_EDGE_CHAIN_BONUS,
    REWARD_HUB_PRIME, REWARD_L_OVERLOAD_EXECUTE, SIM_CASCADE_MAX_PENALTY,
    REWARD_EXPOSURE_GAIN,
    PENALTY_SIM_CASCADE_LOSS,
    CHAIN_REWARD_CAP,
    PENALTY_UNPRIMED_CELLS,
)

NAMED_POLICIES = {
    "random":    random_policy,
    "greedy":    greedy_capture_policy,
    "defensive": defensive_policy,
}

PANEL_W     = 340
PANEL_BG    = (15, 15, 25)
PANEL_ALPHA = 230
COL_POS     = (80, 210, 80)
COL_NEG     = (220, 70, 70)
COL_ZERO    = (120, 120, 140)
COL_WHITE   = (230, 230, 230)
COL_TITLE   = (180, 180, 220)


# ------------------------------------------------------------------
# Headless cascade helper
# ------------------------------------------------------------------

def simulate_headless(g):
    safety = 0
    while g.state == "animating":
        wave = g.get_wave()
        if not wave:
            break
        g.apply_wave(wave)
        safety += 1
        if safety > 10_000:
            break
    return g


# ------------------------------------------------------------------
# Reward breakdown
# ------------------------------------------------------------------

def breakdown(game_post, player, counts_before, counts_after,
              exposure_before, enemy_primed_before,
              cell_snapshot, pos_before, term_reward,
              r_placed=None, c_placed=None, game_pre=None):
    """
    Compute every reward/penalty component for display.

    Parameters
    ----------
    game_post           : Game after move fully resolved (post-cascade)
    player              : player who just moved
    counts_before/after : orb counts [p0, p1] before and after the move
    exposure_before     : count_exposure(game_pre, player) — pre-move
    enemy_primed_before : count_primed(game_pre, enemy) — pre-move (kept for
                          Type 2 display; L-overload execution uses pos_before)
    cell_snapshot       : {(r,c): count} of player's cells before move
    pos_before          : _positional_snapshot before move
    term_reward         : 5.0 if player just won, else 0.0
    r_placed, c_placed  : placement coordinates (for wasted penalty)
    game_pre            : Game snapshot before move (for wasted penalty)
    """
    enemy  = 1 - player
    parts  = {}

    # -- Build current primed sets --
    my_primed    = set()
    enemy_primed = set()
    for r in range(game_post.rows):
        for c in range(game_post.cols):
            cell = game_post.grid[r][c]
            if cell.owner == player and game_post.is_primed(r, c):
                my_primed.add((r, c))
            elif cell.owner == enemy and game_post.is_primed(r, c):
                enemy_primed.add((r, c))

    enemy_cells = {(r, c) for r in range(game_post.rows) for c in range(game_post.cols)
                   if game_post.grid[r][c].owner == enemy}

    exposed_post = {(r, c) for (r, c) in my_primed
                    if any(nb in enemy_primed for nb in game_post.neighbours(r, c))}
    clean        = len(exposed_post) <= exposure_before

    # -- Per-move rewards --
    captured = max(counts_before[enemy] - counts_after[enemy], 0)
    parts["capture"] = REWARD_CLEAN_CAPTURE * captured if (clean and captured) else 0.0

    cascade_gain = max((counts_after[player] - counts_before[player]) - 1, 0)
    parts["cascade"] = REWARD_CASCADE_OWN * cascade_gain

    # Phase boost: mirrors train.py — early-game rewards worth more
    total_orbs  = sum(counts_after)
    phase_boost = max(1.0, 3.0 - total_orbs / 10.0)

    # -- Safe primed cells (delta): stepping stone toward chain discovery --
    safe_primes = sum(
        1 for (r, c) in my_primed
        if not any(nb in enemy_primed for nb in game_post.neighbours(r, c))
    )
    parts["safe_prime"] = REWARD_SAFE_PRIME * max(
        safe_primes - pos_before.get('safe_primes', 0), 0) * phase_boost

    # -- Any new primed cell (delta): signal even for unsafe primes --
    all_primes = len(my_primed)
    parts["any_prime"] = REWARD_ANY_PRIME * max(
        all_primes - pos_before.get('all_primes', 0), 0) * phase_boost

    # -- Chain pairs: full current score (capped) if chain grew --
    chain_pairs = sum(
        1 for (r, c) in my_primed
        for nb in game_post.neighbours(r, c) if nb in my_primed
    ) // 2
    parts["chain"] = (REWARD_CHAIN * min(chain_pairs, CHAIN_REWARD_CAP)
                      if chain_pairs > pos_before['chain_pairs'] else 0.0)

    # -- Safe attack (delta): primed adj to unprimed enemy, not exposed --
    attacking = {(r, c) for (r, c) in my_primed
                 if any(nb in enemy_cells for nb in game_post.neighbours(r, c))}
    safe_attacking = {(r, c) for (r, c) in attacking
                      if not any(nb in enemy_primed for nb in game_post.neighbours(r, c))}
    parts["safe_atk"] = REWARD_SAFE_ATTACK * max(
        len(safe_attacking) - pos_before['safe_attack'], 0)

    # -- Attack chain: full current score (capped) if it grew --
    atk_chain = sum(
        REWARD_ATTACK_CHAIN / 2
        for (r, c) in my_primed
        for nb in game_post.neighbours(r, c)
        if nb in my_primed and ((r, c) in attacking or nb in attacking)
    )
    if atk_chain > pos_before['attack_chain_score']:
        parts["atk_chain"] = min(atk_chain, REWARD_ATTACK_CHAIN * CHAIN_REWARD_CAP)
    else:
        parts["atk_chain"] = 0.0

    # -- Corner control (delta): any newly claimed corner cell --
    corners = {(0, 0), (0, game_post.cols-1),
               (game_post.rows-1, 0), (game_post.rows-1, game_post.cols-1)}
    corner_orbs = sum(1 for (r, c) in corners if game_post.grid[r][c].owner == player)
    parts["corner_ctrl"] = REWARD_CORNER_CONTROL * max(
        corner_orbs - pos_before.get('corner_orbs', 0), 0) * phase_boost

    # -- Corner execute: setup existed pre-move and corner has now fired --
    corners_executed = sum(
        1 for (r, c) in pos_before.get('corner_setup_cells', frozenset())
        if game_post.grid[r][c].count != game_post.critical_mass(r, c) - 1
    )
    parts["corner_execute"] = REWARD_CORNER_EXECUTE * corners_executed * phase_boost

    # -- Edge chain bonus (delta) --
    edge_cells = {(r, c) for r in range(game_post.rows) for c in range(game_post.cols)
                  if r == 0 or r == game_post.rows - 1 or c == 0 or c == game_post.cols - 1}
    edge_chain_pairs = sum(
        1 for (r, c) in my_primed
        for nb in game_post.neighbours(r, c)
        if nb in my_primed and ((r, c) in edge_cells or nb in edge_cells)
    ) // 2
    parts["edge_chain"] = REWARD_EDGE_CHAIN_BONUS * max(
        edge_chain_pairs - pos_before.get('edge_chain_pairs', 0), 0)

    # -- Hub prime (delta): >= 3 primed neighbours only (no linear chains) --
    hub_primes = sum(
        1 for (r, c) in my_primed
        if sum(1 for nb in game_post.neighbours(r, c) if nb in my_primed) >= 3
    )
    parts["hub_prime"] = REWARD_HUB_PRIME * max(
        hub_primes - pos_before.get('hub_primes', 0), 0)

    # -- L-overload EXECUTION: config existed pre-move + enemy primed destroyed --
    if pos_before.get('l_overload', 0) > 0:
        ep_destroyed = pos_before.get('enemy_primed_count', 0) - len(enemy_primed)
        parts["l_overload_exec"] = (REWARD_L_OVERLOAD_EXECUTE * ep_destroyed
                                    if ep_destroyed > 0 else 0.0)
    else:
        parts["l_overload_exec"] = 0.0

    # -- TYPE 1: simulated cascade penalty --
    if exposed_post:
        threatening    = {nb for (r, c) in exposed_post
                          for nb in game_post.neighbours(r, c) if nb in enemy_primed}
        total_sim_loss = sum(_simulate_enemy_fire(game_post, er, ec, player)
                             for (er, ec) in threatening)
        parts["sim_casc"] = -min(PENALTY_SIM_CASCADE_LOSS * total_sim_loss, SIM_CASCADE_MAX_PENALTY)
    else:
        parts["sim_casc"] = 0.0

    # -- TYPE 2: exposure resolution reward --
    exposure_after = len(exposed_post)
    if exposure_before > 0 and exposure_after < exposure_before:
        enemy_primed_after = sum(
            1 for r in range(game_post.rows) for c in range(game_post.cols)
            if game_post.grid[r][c].owner == enemy and game_post.is_primed(r, c)
        )
        primed_destroyed  = max(enemy_primed_before - enemy_primed_after, 0)
        parts["exp_gain"] = REWARD_EXPOSURE_GAIN * primed_destroyed
    else:
        parts["exp_gain"] = 0.0

    # -- Unprimed cell penalty (midgame) --
    if total_orbs > 10:
        owned_cells    = sum(1 for r in range(game_post.rows) for c in range(game_post.cols)
                             if game_post.grid[r][c].owner == player)
        owned_unprimed = owned_cells - len(my_primed)
        parts["unprimed"] = (-PENALTY_UNPRIMED_CELLS * (owned_unprimed - 5)
                             if owned_unprimed > 5 else 0.0)
    else:
        parts["unprimed"] = 0.0

    # -- Wasted placement (pre-move, one-time) --
    parts["wasted"] = (
        wasted_placement_penalty(game_pre, player, r_placed, c_placed)
        if game_pre is not None and r_placed is not None
        else 0.0
    )

    parts["TERMINAL"] = term_reward

    return parts


# ------------------------------------------------------------------
# Overlay panel
# ------------------------------------------------------------------

def draw_panel(screen, font_lg, font_sm, player, cell, parts, cumulative, step):
    sw, sh = screen.get_size()
    px     = sw - PANEL_W

    surf = pygame.Surface((PANEL_W, sh), pygame.SRCALPHA)
    surf.fill((*PANEL_BG, PANEL_ALPHA))

    y  = 12
    lh = 22

    def txt(text, colour, fx, fy, font=None):
        img = (font or font_sm).render(text, True, colour)
        surf.blit(img, (fx, fy))

    pname = ["Red", "Blue"][player]
    txt(f"Step {step}  —  P{player} ({pname})", COL_TITLE, 10, y, font_lg)
    y += 28
    txt(f"placed on ({cell[0]}, {cell[1]})", COL_WHITE, 10, y)
    y += lh + 6
    pygame.draw.line(surf, (60, 60, 80), (10, y), (PANEL_W - 10, y))
    y += 8

    txt("Component", (150, 150, 170), 10, y)
    txt("Value",     (150, 150, 170), PANEL_W - 80, y)
    y += lh

    total = 0.0
    for name, val in parts.items():
        col     = COL_ZERO if abs(val) < 1e-6 else (COL_POS if val > 0 else COL_NEG)
        bar_len = min(int(abs(val) / 0.5 * 80), 80)
        bar_col = (40, 100, 40) if val >= 0 else (100, 40, 40)
        if bar_len > 0:
            pygame.draw.rect(surf, bar_col, (10, y + 4, bar_len, 12))
        txt(f"  {name}", col, 10, y)
        txt(f"{val:+.4f}", col, PANEL_W - 75, y)
        total += val
        y     += lh

    y += 4
    pygame.draw.line(surf, (60, 60, 80), (10, y), (PANEL_W - 10, y))
    y += 8

    col = COL_POS if total >= 0 else COL_NEG
    txt("STEP TOTAL", COL_WHITE, 10, y, font_lg)
    txt(f"{total:+.4f}", col, PANEL_W - 75, y, font_lg)
    y += 30

    txt("Cumulative", COL_TITLE, 10, y)
    y += lh
    for p in range(2):
        pn  = ["Red", "Blue"][p]
        col = COL_POS if cumulative[p] >= 0 else COL_NEG
        txt(f"  P{p} ({pn}): {cumulative[p]:+.4f}", col, 10, y)
        y += lh

    y += 10
    pygame.draw.line(surf, (60, 60, 80), (10, y), (PANEL_W - 10, y))
    y += 10
    txt("SPACE / ENTER  next move", (100, 100, 130), 10, y)
    y += lh
    txt("U  undo last move", (100, 100, 130), 10, y)
    y += lh
    txt("R  restart     Esc  quit", (100, 100, 130), 10, y)

    screen.blit(surf, (px, 0))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def load_agent(path, env):
    if path and os.path.exists(path):
        print(f"  Loading agent: {path}")
        return DQNAgent.from_checkpoint(path)
    return None


def get_action(game, env, agent, policy_fn, player, epsilon,
               filter_moves=False, exposure_before=0):
    if policy_fn is not None:
        r, c = policy_fn(game, player)
        return r, c
    state     = env.encode_state(player)
    mask      = env.valid_action_mask(player)
    remaining = mask.copy()
    while True:
        action       = agent.select_action(state, remaining, epsilon)
        r_act, c_act = divmod(action, game.cols)
        risk = _move_risk_score(game, player, r_act, c_act)
        if not filter_moves or risk == 0.0:
            return r_act, c_act
        remaining[action] = False
        if not remaining.any():
            action = agent.select_action(state, mask, epsilon)
            return divmod(action, game.cols)


def seat_label(i, args):
    p = [args.policy0, args.policy1][i]
    c = [args.ckpt0,   args.ckpt1  ][i]
    if p: return f"Policy({p})"
    if c: return os.path.basename(c)
    return "RandomNet"


def run(args):
    pygame.init()
    pygame.display.set_caption("Chain Reaction — Debug Watch")

    rows, cols = args.rows, args.cols
    any_ckpt   = args.ckpt0 or args.ckpt1
    if any_ckpt and os.path.exists(any_ckpt):
        import torch
        meta = torch.load(any_ckpt, map_location="cpu", weights_only=False)
        rows = meta.get("rows", rows)
        cols = meta.get("cols", cols)
        print(f"  Board from checkpoint: {rows}x{cols}")
    else:
        print(f"  Board size: {rows}x{cols}")

    board_w = 760
    screen  = pygame.display.set_mode((board_w + PANEL_W, 760), pygame.RESIZABLE)
    clock   = pygame.time.Clock()
    font_lg = pygame.font.SysFont("consolas", 15, bold=True)
    font_sm = pygame.font.SysFont("consolas", 13)

    env      = ChainReactionEnv(rows=rows, cols=cols, num_players=2)
    agents   = [load_agent(args.ckpt0, env), load_agent(args.ckpt1, env)]
    policies = [NAMED_POLICIES.get(args.policy0), NAMED_POLICIES.get(args.policy1)]

    print(f"\n  P0: {seat_label(0, args)}  vs  P1: {seat_label(1, args)}")
    print(f"  SPACE/ENTER to step  |  U to undo  |  R to restart")
    print(f"  force_greedy={args.force_greedy}  filter_moves={args.filter_moves}\n")

    def new_game():
        g          = Game(2, rows, cols)
        board_surf = pygame.Surface((board_w, screen.get_height()))
        r          = GameRenderer(board_surf, g)
        env.game   = g
        return g, r, board_surf

    game, renderer, board_surf = new_game()

    last_parts    = {}
    last_player   = 0
    last_cell     = (0, 0)
    cumulative    = [0.0, 0.0]
    step          = 0
    animating     = False
    panel_visible = False
    wins          = [0, 0]
    history       = []

    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                if event.key == pygame.K_r:
                    game, renderer, board_surf = new_game()
                    last_parts    = {}
                    last_player   = 0
                    last_cell     = (0, 0)
                    cumulative    = [0.0, 0.0]
                    step          = 0
                    animating     = False
                    panel_visible = False
                    history       = []
                    print("\n  --- New game ---")

                if event.key == pygame.K_u:
                    if history:
                        snap = history.pop()
                        game          = snap['game']
                        env.game      = game
                        board_surf    = pygame.Surface((board_w, screen.get_height()))
                        renderer      = GameRenderer(board_surf, game)
                        last_parts    = snap['parts']
                        last_player   = snap['player']
                        last_cell     = snap['cell']
                        cumulative    = snap['cumulative']
                        step          = snap['step']
                        panel_visible = snap['panel_visible']
                        animating     = False
                        print(f"  [undo] back to step {step}")
                    else:
                        print("  [undo] nothing to undo")

                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if not animating and game.state == "placing":
                        player = game.current_player
                        enemy  = 1 - player

                        exposure_pre_check = count_exposure(game, player)
                        force_fired = (
                            args.force_greedy
                            and exposure_pre_check > 0
                            and _has_big_cluster(game, player)
                        )
                        if force_fired:
                            r, c = greedy_capture_policy(game, player)
                        else:
                            r, c = get_action(game, env, agents[player],
                                              policies[player], player, args.epsilon,
                                              filter_moves=args.filter_moves,
                                              exposure_before=exposure_pre_check)

                        counts_before       = game.orb_counts()
                        exposure_before     = exposure_pre_check
                        enemy_primed_before = count_primed(game, enemy)
                        pos_before          = _positional_snapshot(game, player)
                        cell_snapshot       = {
                            (rr, cc): game.grid[rr][cc].count
                            for rr in range(rows) for cc in range(cols)
                            if game.grid[rr][cc].owner == player
                        }

                        history.append({
                            'game':          copy.deepcopy(game),
                            'parts':         dict(last_parts),
                            'player':        last_player,
                            'cell':          last_cell,
                            'cumulative':    list(cumulative),
                            'step':          step,
                            'panel_visible': panel_visible,
                        })

                        game_pre_snap = copy.deepcopy(game)
                        sim           = copy.deepcopy(game)
                        sim           = simulate_headless(sim)
                        sim.place(r, c)
                        sim           = simulate_headless(sim)
                        counts_after  = sim.orb_counts()
                        done_sim      = (sim.state == "won")
                        term          = 5.0 if (done_sim and sim.winner == player) else 0.0

                        parts = breakdown(
                            sim, player,
                            counts_before, counts_after,
                            exposure_before, enemy_primed_before,
                            cell_snapshot, pos_before, term,
                            r_placed=r, c_placed=c,
                            game_pre=game_pre_snap,
                        )
                        total = sum(parts.values())
                        cumulative[player] += total

                        last_parts    = parts
                        last_player   = player
                        last_cell     = (r, c)
                        step         += 1
                        panel_visible = True

                        game.place(r, c)
                        renderer._phase_timer = 0
                        animating = True

                        fg_tag = "  [FORCE GREEDY]" if force_fired else ""
                        print(f"\nStep {step:3d} | P{player} -> ({r},{c}){fg_tag}")
                        for k, v in parts.items():
                            if abs(v) > 1e-6:
                                bar  = "█" * min(int(abs(v) / 0.05), 20)
                                sign = "+" if v > 0 else ""
                                print(f"  {k:>16s}  {sign}{v:.5f}  {bar}")
                        print(f"  {'TOTAL':>16s}  {total:+.5f}")

                    elif game.state == "won":
                        game, renderer, board_surf = new_game()
                        last_parts    = {}
                        cumulative    = [0.0, 0.0]
                        step          = 0
                        panel_visible = False
                        history       = []
                        print("\n  --- New game ---")

        renderer.update(dt)
        if game.state in ("placing", "won"):
            animating = False

        if game.state == "won":
            w = game.winner
            wins[w] += 1
            print(f"\n  === {seat_label(w, args)} wins! "
                  f"Score: {wins[0]}–{wins[1]} ===")
            print("  Press SPACE for new game.")

        board_surf = pygame.Surface((board_w, screen.get_height()))
        renderer.screen = board_surf
        renderer.draw()
        screen.blit(board_surf, (0, 0))

        if panel_visible and last_parts:
            draw_panel(screen, font_lg, font_sm,
                       last_player, last_cell, last_parts, cumulative, step)
        elif not panel_visible:
            hint = font_lg.render("SPACE / ENTER  to make a move", True, (120, 120, 160))
            screen.blit(hint, (screen.get_width() - PANEL_W + 20,
                                screen.get_height() - 40))

        pygame.display.flip()


def parse_args():
    p = argparse.ArgumentParser(
        description="Visual step-by-step reward debugger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt0",   type=str,   default=None,
                   help="Checkpoint for P0 (agent). Omit to use policy.")
    p.add_argument("--ckpt1",   type=str,   default=None,
                   help="Checkpoint for P1 (agent). Omit to use policy.")
    p.add_argument("--policy0", type=str,   default=None,
                   choices=["random", "greedy", "defensive"],
                   help="Fixed policy for P0 (used if no --ckpt0)")
    p.add_argument("--policy1", type=str,   default=None,
                   choices=["random", "greedy", "defensive"],
                   help="Fixed policy for P1 (used if no --ckpt1)")
    p.add_argument("--rows",    type=int,   default=6)
    p.add_argument("--cols",    type=int,   default=6)
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="Exploration rate for agent seats (0 = fully greedy)")
    p.add_argument("--force_greedy", action="store_true",
                   help="Mirror training force-greedy override")
    p.add_argument("--filter_moves", action="store_true",
                   help="Mirror training risk filter")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
