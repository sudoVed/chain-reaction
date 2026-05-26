# ChainReaction.pyw  —  Double-click launcher (no console window)
#
# Windows associates .pyw files with pythonw.exe automatically when Python
# is installed, so double-clicking this opens the game directly — no CMD
# window, no flashing, no terminal.
#
# If double-clicking does nothing: right-click → Open with → Python (pythonw)

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
os.chdir(_here)
sys.path.insert(0, _here)

import main
try:
    main.main()
except Exception:
    import traceback
    main._log("CRASH (pyw):\n" + traceback.format_exc())
