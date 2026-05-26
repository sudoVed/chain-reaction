"""
launcher_stub.py — Thin .exe stub for Chain Reaction.

This file is compiled into ChainReaction.exe via build_exe.bat.
It does NOT bundle any game code — it simply locates pythonw.exe and
runs main.py from the same directory as the exe.

Edit any .py game file and re-run the exe; changes are reflected immediately.
Only rebuild the exe if this launcher file itself changes (which it never should).
"""

import os
import sys
import subprocess

def main():
    # When frozen by PyInstaller, sys.executable is the .exe itself.
    # The game files live in the same folder as the exe.
    if getattr(sys, "frozen", False):
        here = os.path.dirname(sys.executable)
    else:
        here = os.path.dirname(os.path.abspath(__file__))

    main_py = os.path.join(here, "main.py")

    # Find pythonw.exe — same interpreter that's on PATH.
    # pythonw suppresses the console window, exactly like double-clicking .pyw.
    pythonw = os.path.join(os.path.dirname(sys.executable)
                           if not getattr(sys, "frozen", False)
                           else sys.executable,
                           "..")   # will be overridden below

    # Resolve pythonw.exe relative to the current Python install
    import shutil
    pythonw_path = shutil.which("pythonw.exe") or shutil.which("pythonw")

    if not pythonw_path:
        # Fallback: python.exe (shows a brief console flash but still works)
        pythonw_path = shutil.which("python.exe") or shutil.which("python")

    if not pythonw_path or not os.path.exists(main_py):
        import ctypes
        msg = ""
        if not pythonw_path:
            msg += "pythonw.exe not found on PATH.\nInstall Python and ensure it is on PATH.\n\n"
        if not os.path.exists(main_py):
            msg += f"main.py not found at:\n{main_py}\n\nMove ChainReaction.exe back into the game folder."
        ctypes.windll.user32.MessageBoxW(0, msg.strip(), "Chain Reaction — Launch Error", 0x10)
        sys.exit(1)

    # Launch the game detached — this exe can then exit immediately.
    subprocess.Popen(
        [pythonw_path, main_py],
        cwd=here,
        close_fds=True,
    )


if __name__ == "__main__":
    main()
