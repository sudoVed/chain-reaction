"""
main.py  –  Entry point for Chain Reaction.

Run with:  python main.py
Requires:  pip install pygame   (or pygame-ce for Python 3.14+)

Two game modes
--------------
  Normal (human vs human) : returned by setup when the user clicks "Play".
  VS Model                 : returned when the user clicks "Play with Model"
                             and chooses an opponent (Defensive / Greedy / Smart).
                             Human is always Player 0 (Red); AI is Player 1 (Blue).
"""

import sys
import os
import threading
import pygame

from constants import WINDOW_TITLE, FPS, SETUP_BG
from game import Game
from renderer import SetupScreen, GameRenderer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- startup debug log (writes to debug.txt next to main.py) ----------
import traceback as _tb
_DEBUG_LOG = os.path.join(BASE_DIR, "debug.txt")
def _log(msg: str):
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as _f:
            _f.write(msg + "\n")
            _f.flush()
    except Exception:
        pass
# Clear log on each fresh start
try:
    open(_DEBUG_LOG, "w").close()
except Exception:
    pass
_log("imports done")

WINDOW_W      = 860
WINDOW_H      = 760
AI_MOVE_DELAY = 500   # ms the AI "thinks" before placing (so the human can see)


# BGM playlist — alternates between the two tracks indefinitely.
_BGM_TRACKS = [os.path.join("assets", "bgm.mp3"), os.path.join("assets", "bgm2.mp3")]
_BGM_INDEX  = 0
BGM_END     = pygame.USEREVENT + 1   # fired by pygame when a track finishes


def _start_bgm():
    """
    Start the BGM playlist.  bgm.mp3 and bgm2.mp3 play one after the other,
    looping forever.  Uses a USEREVENT so the main loop can advance the track
    without blocking.  Silently skips if either file is unavailable.
    """
    global _BGM_INDEX
    _BGM_INDEX = 0
    _play_bgm_track(_BGM_INDEX)


def _play_bgm_track(index: int):
    """Load and start one track; set BGM_END to fire when it finishes."""
    try:
        path = os.path.join(BASE_DIR, _BGM_TRACKS[index % len(_BGM_TRACKS)])
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(0.35)
        pygame.mixer.music.set_endevent(BGM_END)
        pygame.mixer.music.play(0)   # play once; BGM_END fires at the end
    except Exception:
        pass


def _advance_bgm():
    """Call from the main event loop when a BGM_END event is received."""
    global _BGM_INDEX
    _BGM_INDEX = (_BGM_INDEX + 1) % len(_BGM_TRACKS)
    _play_bgm_track(_BGM_INDEX)


def _load_ai_with_screen(screen, settings) -> object:
    """
    Load AIPlayer in a background thread while showing a loading screen.

    Keeps the pygame event loop alive so Windows never marks the window as
    'not responding' — loading torch + the smart model can take 2-5 seconds
    on first import.  Returns the constructed AIPlayer instance.
    """
    container = [None]
    exc_box   = [None]

    def _worker():
        try:
            from ai_player import AIPlayer   # import inside thread so it doesn't block
            container[0] = AIPlayer(settings["ai_opponent"],
                                    settings["rows"], settings["cols"])
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    font_big = pygame.font.SysFont("segoeui", 32, bold=True)
    font_sm  = pygame.font.SysFont("segoeui", 18)
    clock    = pygame.time.Clock()
    dots     = 0
    dot_tick = 0

    while t.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        dot_tick += clock.tick(30)
        if dot_tick >= 400:
            dots = (dots + 1) % 4
            dot_tick = 0

        screen.fill(SETUP_BG)
        msg  = font_big.render("Loading model" + "." * dots, True, (200, 210, 255))
        sub  = font_sm.render("Smart AI initialising — just a moment", True, (120, 130, 170))
        screen.blit(msg, msg.get_rect(center=(screen.get_width() // 2,
                                              screen.get_height() // 2 - 20)))
        screen.blit(sub, sub.get_rect(center=(screen.get_width() // 2,
                                              screen.get_height() // 2 + 24)))
        pygame.display.flip()

    t.join()
    if exc_box[0]:
        raise exc_box[0]
    return container[0]


def run_setup(screen) -> dict:
    """Show setup screen; return settings dict when the player confirms."""
    setup = SetupScreen(screen)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.type == BGM_END:
                _advance_bgm()
            result = setup.handle_event(event)
            if result:
                return result
        setup.draw()
        clock.tick(FPS)


def run_game(screen, settings: dict) -> str:
    """
    Run one human-vs-human game session.
    Returns "again" | "menu" | "quit".
    """
    game     = Game(**settings)
    renderer = GameRenderer(screen, game)
    clock    = pygame.time.Clock()

    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == BGM_END:
                _advance_bgm()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_r and game.state == "won":
                    return "again"
            renderer.handle_event(event)

        if renderer.wants_menu:
            return "menu"

        renderer.update(dt)
        renderer.draw()


def run_game_vs_ai(screen, settings: dict, ai) -> str:
    """
    Run one Human (P0 / Red) vs AI (P1 / Blue) game session.

    The AI moves automatically after AI_MOVE_DELAY ms whenever it is P1's
    turn and the game is in the placing state.  Human events are only forwarded
    to the renderer when it is P0's turn, preventing accidental clicks during
    the AI's phase.  The MENU button and M key are always active.

    Returns "again" | "menu" | "quit".
    """
    game_settings = {k: v for k, v in settings.items()
                     if k not in ("ai_opponent",)}
    game     = Game(**game_settings)
    renderer = GameRenderer(screen, game)
    clock    = pygame.time.Clock()
    ai_timer = 0

    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == BGM_END:
                _advance_bgm()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_r and game.state == "won":
                    return "again"
                if event.key == pygame.K_m:
                    return "menu"
            # MENU button click is valid regardless of whose turn it is
            if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                    and renderer._menu_btn_rect
                    and renderer._menu_btn_rect.collidepoint(event.pos)):
                return "menu"
            # All other events only forwarded on the human's turn
            if game.current_player == 0:
                renderer.handle_event(event)

        if renderer.wants_menu:
            return "menu"

        # AI move: trigger after a short delay so the human can follow along
        if game.state == "placing" and game.current_player == 1:
            ai_timer += dt
            if ai_timer >= AI_MOVE_DELAY:
                ai_timer = 0
                r, c = ai.pick_move(game, 1)
                if game.can_place(r, c):
                    game.place(r, c)
                if game.state == "animating":
                        renderer._phase_timer = 0
        else:
            ai_timer = 0   # reset when it becomes the human's turn again

        renderer.update(dt)
        renderer.draw()


def main():
    _log("main() started")
    pygame.mixer.pre_init(44100, -16, 2, 512)   # low-latency audio
    _log("mixer pre_init done")
    pygame.init()
    _log("pygame.init() done")
    pygame.display.set_caption(WINDOW_TITLE)

    # Window icon
    icon_path = os.path.join(BASE_DIR, "assets", "icon.png")
    if os.path.exists(icon_path):
        try:
            icon = pygame.image.load(icon_path)
            pygame.display.set_icon(icon)
        except Exception:
            pass

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
    _log("window created")
    _start_bgm()
    _log("bgm started")

    while True:
        settings = run_setup(screen)

        if "ai_opponent" in settings:
            ai     = _load_ai_with_screen(screen, settings)
            result = run_game_vs_ai(screen, settings, ai)
        else:
            result = run_game(screen, settings)

        if result == "again":
            # Re-enter the inner play loop with same settings (skip setup)
            while True:
                if "ai_opponent" in settings:
                    result = run_game_vs_ai(screen, settings, ai)
                else:
                    result = run_game(screen, settings)
                if result != "again":
                    break

        if result == "quit":
            break
        # result == "menu" → loop back to run_setup

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log("CRASH:\n" + _tb.format_exc())
        raise
