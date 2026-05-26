"""
renderer.py  -  All pygame drawing for Chain Reaction.

Wave animation pipeline
-----------------------
  idle  -> (grab wave from game)
  burst -> ALL cells in the wave flash simultaneously  (BURST_DURATION_MS)
  flying -> ALL orbs from ALL cells fly to their neighbours simultaneously
            (FLY_DURATION_MS)
        -> game.apply_wave(wave) updates grid state
        -> back to idle; if new wave queued, repeat immediately
"""

import math
import os
import random
import pygame
from dataclasses import dataclass
from constants import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _draw_orb(surf, cx, cy, r, base_col, rim_col, alpha=255):
    r = max(int(r), 3)
    orb_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
    pygame.draw.circle(orb_surf, (*rim_col,  alpha), (r + 1, r + 1), r)
    pygame.draw.circle(orb_surf, (*base_col, alpha), (r + 1, r + 1), int(r * 0.82))
    shine_r   = max(int(r * SHINE_RADIUS_RATIO), 1)
    sx        = int(r + 1 - r * SHINE_OFFSET * 0.8)
    sy        = int(r + 1 - r * SHINE_OFFSET)
    shine_col = _lerp_color(base_col, (255, 255, 255), 0.7)
    pygame.draw.circle(orb_surf, (*shine_col, SHINE_ALPHA), (sx, sy), shine_r)
    surf.blit(orb_surf, (int(cx) - r - 1, int(cy) - r - 1))


def _orb_positions(count, cx, cy, r, angle_deg):
    if count <= 0:
        return []
    sep = 2 * r * ORB_OVERLAP_RATIO
    ang = math.radians(angle_deg)
    if count == 1:
        return [(cx, cy)]
    if count == 2:
        raw = [(-sep / 2, 0), (sep / 2, 0)]
    else:
        h = sep * math.sqrt(3) / 2
        raw = [(0, -h * 2 / 3), (-sep / 2, h / 3), (sep / 2, h / 3)]
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    return [(cx + ox * cos_a - oy * sin_a,
             cy + ox * sin_a + oy * cos_a)
            for ox, oy in raw]


def _draw_panel(surf, rect, bg=CYBER_PANEL, border=CYBER_BORDER_DIM,
                radius=10, border_w=1):
    pygame.draw.rect(surf, bg,     rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, border_w, border_radius=radius)


def _draw_cyber_btn(surf, rect, label, font,
                    primary=False, hovered=False, disabled=False):
    if disabled:
        bg     = (8, 10, 20)
        border = CYBER_BORDER_DIM
        tcol   = CYBER_TEXT_DIM
    elif primary:
        bg     = _lerp_color(CYBER_ACCENT, (0, 0, 0), 0.72 if not hovered else 0.58)
        border = CYBER_ACCENT
        tcol   = (240, 255, 255)
    else:
        bg     = CYBER_BTN_HOV if hovered else CYBER_BTN
        border = CYBER_BORDER   if hovered else CYBER_BORDER_DIM
        tcol   = CYBER_ACCENT   if hovered else CYBER_TEXT

    pygame.draw.rect(surf, bg,     rect, border_radius=8)
    pygame.draw.rect(surf, border, rect, 1, border_radius=8)
    txt = font.render(label, True, tcol)
    surf.blit(txt, txt.get_rect(center=rect.center))


def _draw_brush_stroke(surf, p0, p1, col, w, curve_side=1):
    """
    Curved, tapering brush/marker stroke.

    Routes the stroke through two slightly offset control points to create
    a gentle bow, and uses narrower widths at the endpoints so the stroke
    tapers naturally at both ends. No randomness; stable across frames.

    curve_side: +1 or -1 controls which side of the stroke axis bows outward.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    ln = math.hypot(dx, dy)
    if ln < 1:
        return
    nx, ny = dx / ln, dy / ln
    px, py = -ny * curve_side, nx * curve_side   # perpendicular

    # Two inner control points bowing slightly off the straight path
    c1 = (p0[0] + dx * 0.30 + px * 3.8,
          p0[1] + dy * 0.30 + py * 3.8)
    c2 = (p0[0] + dx * 0.70 + px * 3.2,
          p0[1] + dy * 0.70 + py * 3.2)

    pts    = [p0, c1, c2, p1]
    widths = [max(2, w - 5), w, w, max(2, w - 4)]   # taper at ends

    for i in range(len(pts) - 1):
        a = (int(pts[i][0]),     int(pts[i][1]))
        b = (int(pts[i + 1][0]), int(pts[i + 1][1]))
        pygame.draw.line(surf, col, a, b, widths[i])


def _draw_select_x(surf, cx, cy, r, col):
    """
    Hand-drawn prohibition X through a circle.

    Deliberately imperfect:
      - Top-left and top-right arms are visibly shorter than their opposites.
      - Each stroke deviates slightly from 45 degrees.
      - Strokes are curved and taper at the ends via _draw_brush_stroke.

    ext and w preserve the user-set values — do not alter them.
    """
    ext = int(r * 0.9)
    w   = max(10, int(r * 0.30))

    # '\' stroke — TL arm shorter + slight off-axis; bows up-right
    tl = (cx - int(ext * 0.73), cy - int(ext * 0.84))
    br = (cx + int(ext * 1.04), cy + int(ext * 0.97))
    _draw_brush_stroke(surf, tl, br, col, w, curve_side=1)

    # '/' stroke — TR arm extends beyond circle; curves upward; slightly thicker
    tr = (cx + int(ext * 1.08), cy - int(ext * 0.98))
    bl = (cx - int(ext * 1.02), cy + int(ext * 1.03))
    _draw_brush_stroke(surf, tr, bl, col, w + 3, curve_side=1)

    # Circle drawn on top so the X visually crosses through the ring
    pygame.draw.circle(surf, col, (cx, cy), r, 3)


# ---------------------------------------------------------------------------
# Flying orb
# ---------------------------------------------------------------------------

@dataclass
class FlyingOrb:
    sx: float; sy: float
    dx: float; dy: float
    owner: int

    def pos(self, t: float):
        s = _smoothstep(t)
        return (self.sx + (self.dx - self.sx) * s,
                self.sy + (self.dy - self.sy) * s)


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

class SetupScreen:
    """
    Pre-game configuration screen.

    Two modes:
      "setup"        - player count + grid size selectors + two bottom buttons.
      "model_select" - choose AI opponent with card UI and prohibition-X mark.

    draw() refreshes self.w/h from the surface every frame for resize support.
    handle_event() returns a settings dict on confirm, else None.
    """

    BTN_W, BTN_H = 60, 46
    GAP = 14

    _MODELS = [
        ("defensive", "Scared", "Safe & tactical"),
        ("greedy",    "Ooga Booga",    "Aggressive chains"),
        ("smart",     "AI-nstein",     "Trained AI"),
    ]
    _CARD_W   = 180
    _CARD_H   = 130
    _CIRCLE_R = 34

    def __init__(self, screen):
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.selected_players = 2
        self.selected_grid    = 6
        self._font_title = pygame.font.SysFont("segoeui", 46, bold=True)
        self._font_label = pygame.font.SysFont("segoeui", 20)
        self._font_btn   = pygame.font.SysFont("segoeui", 17, bold=True)
        self._font_start = pygame.font.SysFont("segoeui", 22, bold=True)
        self._font_desc  = pygame.font.SysFont("segoeui", 14)
        self._hovered    = None
        self._mode           = "setup"
        self._selected_model = "smart"

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self._hit_test(event.pos)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit = self._hit_test(event.pos)
            if hit:
                kind, val = hit
                if self._mode == "setup":
                    if kind == "players":
                        self.selected_players = val
                    elif kind == "grid":
                        self.selected_grid = val
                    elif kind == "play":
                        return {"num_players": self.selected_players,
                                "rows":        self.selected_grid,
                                "cols":        self.selected_grid}
                    elif kind == "play_with_model":
                        self._mode = "model_select"
                elif self._mode == "model_select":
                    if kind == "model":
                        self._selected_model = val
                    elif kind == "back":
                        self._mode = "setup"
                    elif kind == "start_model":
                        return {"num_players": 2,
                                "rows":        self.selected_grid,
                                "cols":        self.selected_grid,
                                "ai_opponent": self._selected_model}
        return None

    def draw(self):
        self.w, self.h = self.screen.get_size()
        self.screen.fill(SETUP_BG)
        self._draw_title()
        if self._mode == "setup":
            self._draw_section("Number of Players", PLAYER_OPTIONS,
                               self.selected_players, "players",
                               int(self.h * 0.37))
            self._draw_section("Grid Size", GRID_OPTIONS,
                               self.selected_grid, "grid",
                               int(self.h * 0.57))
            self._draw_setup_buttons()
        else:
            self._draw_model_select()
        pygame.display.flip()

    def _draw_title(self):
        glow = self._font_title.render("Chain Reaction", True, (0, 80, 100))
        self.screen.blit(glow, glow.get_rect(
            center=(self.w // 2 + 2, int(self.h * 0.13) + 2)))
        surf = self._font_title.render("Chain Reaction", True, CYBER_ACCENT)
        self.screen.blit(surf, surf.get_rect(
            center=(self.w // 2, int(self.h * 0.13))))

    def _draw_section(self, label, options, selected, kind, top_y):
        n       = len(options)
        total_w = n * (self.BTN_W + self.GAP) - self.GAP
        pad     = 18
        panel_w = total_w + pad * 2
        panel_h = self.BTN_H + 48
        panel_x = self.w // 2 - panel_w // 2
        _draw_panel(self.screen,
                    pygame.Rect(panel_x, top_y - 32, panel_w, panel_h),
                    bg=CYBER_PANEL, border=CYBER_BORDER_DIM, radius=10)

        lsurf = self._font_label.render(label, True, CYBER_TEXT_DIM)
        self.screen.blit(lsurf, lsurf.get_rect(
            center=(self.w // 2, top_y - 14)))

        sx = self.w // 2 - total_w // 2
        for i, opt in enumerate(options):
            rect   = pygame.Rect(sx + i * (self.BTN_W + self.GAP),
                                 top_y + 8, self.BTN_W, self.BTN_H)
            is_sel = (opt == selected)
            is_hov = (self._hovered == (kind, opt))

            if is_sel:
                bg     = _lerp_color(CYBER_ACCENT, (0, 0, 0), 0.78)
                border = CYBER_ACCENT
                tcol   = CYBER_ACCENT
            elif is_hov:
                bg     = CYBER_BTN_HOV
                border = CYBER_BORDER_DIM
                tcol   = CYBER_TEXT
            else:
                bg     = CYBER_BTN
                border = CYBER_BORDER_DIM
                tcol   = CYBER_TEXT_DIM

            pygame.draw.rect(self.screen, bg,     rect, border_radius=7)
            pygame.draw.rect(self.screen, border, rect,
                             2 if is_sel else 1, border_radius=7)
            txt = self._font_btn.render(str(opt), True, tcol)
            self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_setup_buttons(self):
        btn_gap = 24
        btn_w   = 190
        btn_h   = 54
        total_w = btn_w * 2 + btn_gap
        bx      = self.w // 2 - total_w // 2
        by      = int(self.h * 0.82)
        left_rect  = pygame.Rect(bx,                   by, btn_w, btn_h)
        right_rect = pygame.Rect(bx + btn_w + btn_gap, by, btn_w, btn_h)
        _draw_cyber_btn(self.screen, left_rect,  "Play with Model",
                        self._font_btn, primary=False,
                        hovered=(self._hovered == ("play_with_model", 0)))
        _draw_cyber_btn(self.screen, right_rect, "Play",
                        self._font_start, primary=True,
                        hovered=(self._hovered == ("play", 0)))

    def _draw_model_select(self):
        sub = self._font_label.render("Choose Your Opponent", True, CYBER_TEXT_DIM)
        self.screen.blit(sub, sub.get_rect(
            center=(self.w // 2, int(self.h * 0.26))))

        gap     = max(self._CARD_W + 20, self.w // 4)
        cy      = int(self.h * 0.52)
        cx_list = [self.w // 2 - gap, self.w // 2, self.w // 2 + gap]

        for i, (key, label, desc) in enumerate(self._MODELS):
            self._draw_model_card(cx_list[i], cy, label, desc,
                                  selected=(self._selected_model == key))

        btn_gap = 24
        btn_w   = 170
        btn_h   = 54
        total_w = btn_w * 2 + btn_gap
        bx      = self.w // 2 - total_w // 2
        by      = int(self.h * 0.82)
        back_rect  = pygame.Rect(bx,                   by, btn_w, btn_h)
        start_rect = pygame.Rect(bx + btn_w + btn_gap, by, btn_w, btn_h)
        _draw_cyber_btn(self.screen, back_rect,  "Back",  self._font_start,
                        primary=False,
                        hovered=(self._hovered == ("back",        0)))
        _draw_cyber_btn(self.screen, start_rect, "Start", self._font_start,
                        primary=True,
                        hovered=(self._hovered == ("start_model", 0)))

    def _draw_model_card(self, cx, cy, label, desc, selected):
        cw, ch = self._CARD_W, self._CARD_H
        rect   = pygame.Rect(cx - cw // 2, cy - ch // 2, cw, ch)
        bg     = _lerp_color(CYBER_ACCENT, (0, 0, 0), 0.88) if selected else CYBER_PANEL
        border = CYBER_ACCENT if selected else CYBER_BORDER_DIM
        _draw_panel(self.screen, rect, bg=bg, border=border,
                    radius=10, border_w=2 if selected else 1)

        circle_cy  = cy - 16
        r          = self._CIRCLE_R
        circle_col = CYBER_ACCENT if selected else CYBER_BORDER_DIM
        pygame.draw.circle(self.screen, circle_col, (cx, circle_cy), r, 2)
        if selected:
            _draw_select_x(self.screen, cx, circle_cy, r, CYBER_SELECT_X)

        lcol  = CYBER_ACCENT if selected else CYBER_TEXT
        lsurf = self._font_label.render(label, True, lcol)
        self.screen.blit(lsurf, lsurf.get_rect(center=(cx, cy + ch // 2 - 34)))

        dsurf = self._font_desc.render(desc, True, CYBER_TEXT_DIM)
        self.screen.blit(dsurf, dsurf.get_rect(center=(cx, cy + ch // 2 - 14)))

    def _hit_test(self, pos):
        x, y = pos
        w, h = self.screen.get_size()

        if self._mode == "setup":
            top_y   = int(h * 0.37)
            total_w = len(PLAYER_OPTIONS) * (self.BTN_W + self.GAP) - self.GAP
            sx = w // 2 - total_w // 2
            for i, opt in enumerate(PLAYER_OPTIONS):
                rect = pygame.Rect(sx + i * (self.BTN_W + self.GAP),
                                   top_y + 8, self.BTN_W, self.BTN_H)
                if rect.collidepoint(x, y):
                    return ("players", opt)

            top_y   = int(h * 0.57)
            total_w = len(GRID_OPTIONS) * (self.BTN_W + self.GAP) - self.GAP
            sx = w // 2 - total_w // 2
            for i, opt in enumerate(GRID_OPTIONS):
                rect = pygame.Rect(sx + i * (self.BTN_W + self.GAP),
                                   top_y + 8, self.BTN_W, self.BTN_H)
                if rect.collidepoint(x, y):
                    return ("grid", opt)

            btn_gap = 24
            btn_w   = 190
            btn_h   = 54
            bx      = w // 2 - (btn_w * 2 + btn_gap) // 2
            by      = int(h * 0.82)
            if pygame.Rect(bx,                   by, btn_w, btn_h).collidepoint(x, y):
                return ("play_with_model", 0)
            if pygame.Rect(bx + btn_w + btn_gap, by, btn_w, btn_h).collidepoint(x, y):
                return ("play", 0)

        else:
            gap     = max(self._CARD_W + 20, w // 4)
            cy      = int(h * 0.52)
            cx_list = [w // 2 - gap, w // 2, w // 2 + gap]
            cw, ch  = self._CARD_W, self._CARD_H
            for i, (key, _, _) in enumerate(self._MODELS):
                rect = pygame.Rect(cx_list[i] - cw // 2, cy - ch // 2, cw, ch)
                if rect.collidepoint(x, y):
                    return ("model", key)

            btn_gap = 24
            btn_w   = 170
            btn_h   = 54
            bx      = w // 2 - (btn_w * 2 + btn_gap) // 2
            by      = int(h * 0.82)
            if pygame.Rect(bx,                   by, btn_w, btn_h).collidepoint(x, y):
                return ("back", 0)
            if pygame.Rect(bx + btn_w + btn_gap, by, btn_w, btn_h).collidepoint(x, y):
                return ("start_model", 0)

        return None


# ---------------------------------------------------------------------------
# Game renderer
# ---------------------------------------------------------------------------

class GameRenderer:
    """Draws the board and drives wave-based explosion animations."""

    def __init__(self, screen, game):
        self.screen = screen
        self.game   = game
        self.w, self.h = screen.get_size()
        self.cs = 1
        self.ox = 0
        self.oy = UI_HEIGHT

        self._font_ui  = pygame.font.SysFont("segoeui", 17, bold=True)
        self._font_win = pygame.font.SysFont("segoeui", 56, bold=True)
        self._font_sub = pygame.font.SysFont("segoeui", 22)

        self.spin_angle = 0.0

        self._phase       = "idle"
        self._phase_timer = 0

        self._current_wave = []
        self._wave_owners  = {}
        self._hiding       = set()
        self._flying_orbs  = []

        self._hover = None

        self._undo_btn_rect = None
        self._menu_btn_rect = None
        self.wants_menu     = False

        self._pop_sound = self._load_sound(os.path.join("assets", "pop.mp3"), 0.7)

        # Sparkle / star background
        self._sparkles       = []   # [x, y, born_ms, duration_ms, size]
        self._total_ms       = 0
        self._sparkle_accum  = 0
        self._sparkle_interval = 340

        self._layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _layout(self):
        self.w, self.h = self.screen.get_size()
        avail_w = self.w
        avail_h = self.h - UI_HEIGHT
        cs = min(avail_w // self.game.cols,
                 avail_h // self.game.rows,
                 CELL_SIZE)
        self.cs = max(cs, 1)
        self.ox = (avail_w  - self.cs * self.game.cols) // 2
        self.oy = UI_HEIGHT + (avail_h - self.cs * self.game.rows) // 2

    # ------------------------------------------------------------------
    # Sound
    # ------------------------------------------------------------------

    def _load_sound(self, filename, volume=1.0):
        path = os.path.join(BASE_DIR, filename)
        try:
            snd = pygame.mixer.Sound(path)
            snd.set_volume(volume)
            return snd
        except Exception:
            return None

    def _play_pop(self):
        if self._pop_sound:
            self._pop_sound.play()

    # ------------------------------------------------------------------
    # Cell helpers
    # ------------------------------------------------------------------

    def _cell_center(self, r, c):
        return (self.ox + c * self.cs + self.cs / 2,
                self.oy + r * self.cs + self.cs / 2)

    def _cell_at(self, pos):
        x, y = pos
        if self.cs < 1:
            return None
        c = (x - self.ox) // self.cs
        r = (y - self.oy) // self.cs
        if 0 <= r < self.game.rows and 0 <= c < self.game.cols:
            return (r, c)
        return None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt_ms):
        self.spin_angle = (self.spin_angle + SPIN_SPEED) % 360
        self._tick_sparkles(dt_ms)

        if self.game.state != "animating":
            return

        self._phase_timer += dt_ms

        if self._phase == "idle":
            if self._phase_timer >= EXPLODE_DELAY_MS:
                self._phase_timer = 0
                wave = self.game.get_wave()
                if wave:
                    self._begin_burst(wave)

        elif self._phase == "burst":
            if self._phase_timer >= BURST_DURATION_MS:
                self._phase_timer = 0
                self._begin_flying()

        elif self._phase == "flying":
            if self._phase_timer >= FLY_DURATION_MS:
                self._phase_timer = 0
                self._finish_wave()

    def _begin_burst(self, wave):
        committed = self.game.committed_explosions
        self._current_wave = [
            (r, c) for r, c in wave
            if self.game.grid[r][c].count >= self.game.critical_mass(r, c)
            or (r, c) in committed
        ]
        if not self._current_wave:
            self.game.apply_wave(wave)
            self._phase = "idle"
            return
        self._wave_owners = {
            (r, c): self.game.grid[r][c].owner
            for r, c in self._current_wave
        }
        self._hiding = set(self._current_wave)
        self._phase  = "burst"
        self._play_pop()

    def _begin_flying(self):
        orb_r = self.cs * ORB_RADIUS_RATIO * 0.85
        self._flying_orbs = []
        for r, c in self._current_wave:
            owner = self._wave_owners.get((r, c), -1)
            if owner < 0:
                continue
            sx, sy = self._cell_center(r, c)
            for nr, nc in self.game.neighbours(r, c):
                dx, dy = self._cell_center(nr, nc)
                self._flying_orbs.append(FlyingOrb(sx, sy, dx, dy, owner))
        self._phase = "flying"

    def _finish_wave(self):
        self.game.apply_wave(self._current_wave)
        self._current_wave = []
        self._wave_owners  = {}
        self._hiding       = set()
        self._flying_orbs  = []
        self._phase        = "idle"
        self._phase_timer  = 0

    # ------------------------------------------------------------------
    # Sparkle background
    # ------------------------------------------------------------------

    def _tick_sparkles(self, dt_ms):
        self._total_ms += dt_ms
        self._sparkles = [s for s in self._sparkles
                          if self._total_ms - s[2] < s[3]]
        self._sparkle_accum += dt_ms
        while self._sparkle_accum >= self._sparkle_interval:
            self._sparkle_accum -= self._sparkle_interval
            self._try_spawn_sparkle()

    def _try_spawn_sparkle(self):
        grid_l = self.ox
        grid_r = self.ox + self.game.cols * self.cs
        grid_t = self.oy
        grid_b = self.oy + self.game.rows * self.cs
        margin = 6
        for _ in range(12):
            x = random.randint(margin, max(margin + 1, self.w - margin))
            y = random.randint(UI_HEIGHT + margin,
                               max(UI_HEIGHT + margin + 1, self.h - margin))
            if not (grid_l <= x <= grid_r and grid_t <= y <= grid_b):
                dur  = random.randint(700, 1100)
                size = random.uniform(2.0, 5.5)
                self._sparkles.append([x, y, self._total_ms, dur, size])
                self._sparkle_interval = random.randint(260, 440)
                return

    def _draw_sparkles(self):
        for sx, sy, born, duration, size in self._sparkles:
            age = self._total_ms - born
            t   = max(0.0, min(1.0, age / duration))
            if t < 0.25:
                alpha = t / 0.25
            elif t > 0.65:
                alpha = 1.0 - (t - 0.65) / 0.35
            else:
                alpha = 1.0
            a = int(max(0.0, min(1.0, alpha)) * 220)
            if a < 4:
                continue
            arm  = int(size * 2.6)
            darm = int(size * 1.3)
            pad  = arm + 2
            srf  = pygame.Surface((pad * 2, pad * 2), pygame.SRCALPHA)
            scx  = scy = pad
            col  = (210, 240, 255, a)
            lw   = max(1, int(size * 0.45))
            pygame.draw.line(srf, col, (scx, scy - arm), (scx, scy + arm), lw)
            pygame.draw.line(srf, col, (scx - arm, scy), (scx + arm, scy), lw)
            pygame.draw.line(srf, col,
                             (scx - darm, scy - darm), (scx + darm, scy + darm),
                             max(1, lw - 1))
            pygame.draw.line(srf, col,
                             (scx + darm, scy - darm), (scx - darm, scy + darm),
                             max(1, lw - 1))
            pygame.draw.circle(srf, col, (scx, scy), max(1, int(size * 0.5)))
            self.screen.blit(srf, (sx - pad, sy - pad))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hover = self._cell_at(event.pos)

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_z, pygame.K_u):
                self._try_undo()
            if event.key == pygame.K_m:
                self.wants_menu = True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if (self._menu_btn_rect
                    and self._menu_btn_rect.collidepoint(event.pos)):
                self.wants_menu = True
                return
            if (self._undo_btn_rect
                    and self._undo_btn_rect.collidepoint(event.pos)):
                self._try_undo()
                return
            cell = self._cell_at(event.pos)
            if cell:
                placed = self.game.place(*cell)
                if placed and self.game.state == "animating":
                    self._phase_timer = 0

    def _try_undo(self):
        if self.game.state != "placing":
            return
        if self.game.undo():
            self._phase       = "idle"
            self._phase_timer = 0
            self._current_wave = []
            self._wave_owners  = {}
            self._hiding       = set()
            self._flying_orbs  = []

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self):
        self._layout()
        self.screen.fill(BOARD_BG)
        self._draw_sparkles()
        self._draw_ui_bar()
        self._draw_grid()
        self._draw_cells()
        self._draw_flying_orbs()
        if self._phase == "burst":
            self._draw_bursts()
        if self.game.state == "won":
            self._draw_win_screen()
        pygame.display.flip()

    def _draw_ui_bar(self):
        g = self.game
        pygame.draw.rect(self.screen, (8, 8, 20), (0, 0, self.w, UI_HEIGHT))
        pygame.draw.line(self.screen, CYBER_BORDER_DIM,
                         (0, UI_HEIGHT - 1), (self.w, UI_HEIGHT - 1))

        if g.state in ("placing", "animating"):
            col  = PLAYER_COLORS[g.current_player]["base"]
            name = PLAYER_COLORS[g.current_player]["name"]
            pygame.draw.rect(self.screen, col, (0, 0, 3, UI_HEIGHT))
            pygame.draw.circle(self.screen, col, (22, UI_HEIGHT // 2), 10)
            txt = self._font_ui.render(name + "'s turn", True, (215, 225, 255))
            self.screen.blit(txt, (38, UI_HEIGHT // 2 - txt.get_height() // 2))

        mbtn_w, mbtn_h = 76, 30
        mbtn = pygame.Rect(self.w - mbtn_w - 10,
                           UI_HEIGHT // 2 - mbtn_h // 2, mbtn_w, mbtn_h)
        self._menu_btn_rect = mbtn
        _draw_cyber_btn(self.screen, mbtn, "MENU", self._font_ui,
                        primary=False,
                        hovered=mbtn.collidepoint(pygame.mouse.get_pos()))

        counts = g.orb_counts()
        rx = mbtn.left - 8
        for p in range(g.num_players - 1, -1, -1):
            if not g.alive[p]:
                continue
            col  = PLAYER_COLORS[p]["base"]
            ctxt = self._font_ui.render(str(counts[p]), True, col)
            rx  -= ctxt.get_width()
            self.screen.blit(ctxt, (rx, UI_HEIGHT // 2 - ctxt.get_height() // 2))
            pygame.draw.circle(self.screen, col, (rx - 12, UI_HEIGHT // 2), 7)
            rx -= 26

        can_undo = g.state == "placing" and bool(g.history)
        bw, bh   = 104, 30
        btn_rect = pygame.Rect(self.w // 2 - bw // 2,
                               UI_HEIGHT // 2 - bh // 2, bw, bh)
        self._undo_btn_rect = btn_rect
        _draw_cyber_btn(self.screen, btn_rect, "<- UNDO", self._font_ui,
                        primary=False, disabled=not can_undo)

    def _draw_grid(self):
        cs, ox, oy = self.cs, self.ox, self.oy
        g = self.game

        corners = [(0, 0), (0, g.cols - 1), (g.rows - 1, 0), (g.rows - 1, g.cols - 1)]
        for r, c in corners:
            hs = pygame.Surface((cs - 1, cs - 1), pygame.SRCALPHA)
            hs.fill((0, 200, 220, 10))
            self.screen.blit(hs, (ox + c * cs + 1, oy + r * cs + 1))

        for r in range(g.rows + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (ox, oy + r * cs), (ox + g.cols * cs, oy + r * cs))
        for c in range(g.cols + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (ox + c * cs, oy), (ox + c * cs, oy + g.rows * cs))

        if self._hover and self.game.state == "placing":
            r, c = self._hover
            if self.game.can_place(r, c):
                pcol = PLAYER_COLORS[self.game.current_player]["base"]
                hl   = pygame.Surface((cs - 1, cs - 1), pygame.SRCALPHA)
                hl.fill((*pcol, 40))
                self.screen.blit(hl, (ox + c * cs + 1, oy + r * cs + 1))

    def _draw_cells(self):
        for r in range(self.game.rows):
            for c in range(self.game.cols):
                if (r, c) in self._hiding:
                    continue
                cell = self.game.grid[r][c]
                if cell.is_empty() or cell.count == 0:
                    continue
                self._draw_cell_orbs(r, c, cell)

    def _draw_cell_orbs(self, r, c, cell):
        cx, cy = self._cell_center(r, c)
        orb_r  = self.cs * ORB_RADIUS_RATIO
        primed = self.game.is_primed(r, c)
        angle  = self.spin_angle if primed else 0.0
        pcol   = PLAYER_COLORS[cell.owner]
        for px, py in _orb_positions(cell.count, cx, cy, orb_r, angle):
            _draw_orb(self.screen, px, py, orb_r, pcol["base"], pcol["rim"])

    def _draw_bursts(self):
        t = self._phase_timer / BURST_DURATION_MS
        for r, c in self._current_wave:
            self._draw_one_burst(r, c, t)

    def _draw_one_burst(self, r, c, t):
        cx, cy = self._cell_center(r, c)
        cs     = self.cs
        owner  = self._wave_owners.get((r, c), -1)
        col    = PLAYER_COLORS[owner]["trail"] if owner >= 0 else (255, 255, 255)

        ring_r = int(cs * 0.25 + cs * 0.35 * t)
        alpha  = int(255 * (1 - t))
        rs = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(rs, (*col, alpha), (ring_r + 2, ring_r + 2), ring_r, 3)
        self.screen.blit(rs, (cx - ring_r - 2, cy - ring_r - 2))

        flash_r = int(cs * 0.30 * (1 - t))
        if flash_r > 1:
            fs = pygame.Surface((flash_r * 2, flash_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(fs, (255, 255, 255, int(220 * (1 - t))),
                               (flash_r, flash_r), flash_r)
            self.screen.blit(fs, (cx - flash_r, cy - flash_r))

    def _draw_flying_orbs(self):
        if self._phase != "flying" or not self._flying_orbs:
            return
        t     = self._phase_timer / FLY_DURATION_MS
        orb_r = self.cs * ORB_RADIUS_RATIO * 0.85
        for orb in self._flying_orbs:
            px, py = orb.pos(t)
            pcol   = PLAYER_COLORS[orb.owner]
            _draw_orb(self.screen, px, py, orb_r, pcol["base"], pcol["rim"])

    def _draw_win_screen(self):
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((4, 4, 14, 215))
        self.screen.blit(overlay, (0, 0))

        p    = self.game.winner
        col  = PLAYER_COLORS[p]["base"]
        name = PLAYER_COLORS[p]["name"]

        pw, ph   = min(520, self.w - 40), 220
        px       = self.w  // 2 - pw // 2
        py       = self.h  // 2 - ph // 2
        panel_rect = pygame.Rect(px, py, pw, ph)
        _draw_panel(self.screen, panel_rect,
                    bg=(6, 8, 20), border=col, radius=12, border_w=2)

        sl = pygame.Surface((pw, ph), pygame.SRCALPHA)
        for sy in range(0, ph, 3):
            pygame.draw.line(sl, (0, 0, 0, 22), (0, sy), (pw, sy))
        self.screen.blit(sl, (px, py))

        txt = self._font_win.render(name + " wins!", True, col)
        self.screen.blit(txt, txt.get_rect(
            center=(self.w // 2, self.h // 2 - 32)))

        sub1 = self._font_sub.render("R - play again  .  M - menu",
                                     True, (160, 165, 205))
        sub2 = self._font_sub.render("Esc - quit",
                                     True, (110, 115, 155))
        self.screen.blit(sub1, sub1.get_rect(
            center=(self.w // 2, self.h // 2 + 38)))
        self.screen.blit(sub2, sub2.get_rect(
            center=(self.w // 2, self.h // 2 + 72)))
