import pygame

# --- Window ---
WINDOW_TITLE = "Chain Reaction"
FPS = 60

# --- Cyber color palette ---
CYBER_BG         = (6,   6,  14)    # near-black blue-tinted background
CYBER_PANEL      = (10,  12,  26)   # slightly lighter panel fill
CYBER_BORDER     = (0,  200, 220)   # neon cyan border (active)
CYBER_BORDER_DIM = (0,   70,  90)   # dimmed cyan border (inactive)
CYBER_ACCENT     = (0,  240, 255)   # electric cyan — primary accent
CYBER_TEXT       = (200, 220, 255)  # main text
CYBER_TEXT_DIM   = (90, 110, 150)   # dimmed / secondary text
CYBER_BTN        = (10,  14,  30)   # button fill (normal)
CYBER_BTN_HOV    = (18,  24,  50)   # button fill (hover)
CYBER_SELECT_X   = ( 40, 255,  70)  # neon green for model-select X mark

# --- Setup screen (aliases) ---
SETUP_BG      = CYBER_BG
SETUP_TEXT    = CYBER_TEXT
SETUP_ACCENT  = CYBER_ACCENT
SETUP_BTN     = CYBER_BTN
SETUP_BTN_HOV = CYBER_BTN_HOV

# --- Game board ---
BOARD_BG       = CYBER_BG
GRID_LINE      = (0,  65, 85)      # dim neon-cyan grid lines
CELL_SIZE      = 72
CELL_PAD       = 6
UI_HEIGHT      = 56

# --- Animation timing ---
SPIN_SPEED        = 1.2
EXPLODE_DELAY_MS  = 80
BURST_DURATION_MS = 120
FLY_DURATION_MS   = 380

# --- Orb appearance ---
ORB_RADIUS_RATIO   = 0.22
ORB_OVERLAP_RATIO  = 0.55
SHINE_OFFSET       = 0.28
SHINE_RADIUS_RATIO = 0.35
SHINE_ALPHA        = 160

# --- Player colours ---
PLAYER_COLORS = [
    {"name": "Red",     "base": (255,  50,  70), "rim": (160, 10,  30), "trail": (255, 130, 140)},
    {"name": "Blue",    "base": ( 40, 160, 255), "rim": ( 10, 70, 180), "trail": (120, 200, 255)},
    {"name": "Green",   "base": ( 50, 220,  90), "rim": ( 10,120,  40), "trail": (130, 255, 160)},
    {"name": "Yellow",  "base": (255, 210,  30), "rim": (160,120,   0), "trail": (255, 240, 140)},
    {"name": "Magenta", "base": (220,  40, 210), "rim": (130, 10, 120), "trail": (255, 140, 250)},
    {"name": "Cyan",    "base": (  0, 220, 240), "rim": (  0,110, 140), "trail": (140, 255, 255)},
]

# Grid size options shown in setup
GRID_OPTIONS   = [5, 6, 7, 8, 9, 10, 11, 12]
PLAYER_OPTIONS = [2, 3, 4, 5, 6]
