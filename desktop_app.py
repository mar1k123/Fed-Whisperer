"""
Fraud Ring Finder — Native PC Desktop Application Launcher (Python).

Launches the full interactive Fraud Ring Finder desktop suite in a native OS window
on Windows, macOS, or Linux using `pywebview` (if installed) or Chromium/Edge `--app`
frameless native window mode, backed by either the local Express/Vite server or Streamlit.

Usage:
    python desktop_app.py
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def find_free_port(preferred: int = 3000) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def open_native_os_window(url: str, title: str = "Fraud Ring Finder — Graph ML Workbench") -> bool:
    try:
        import webview  # type: ignore[import-not-found]

        webview.create_window(
            title,
            url,
            width=1480,
            height=920,
            min_size=(1100, 720),
            text_select=True,
        )
        webview.start()
        return True
    except ImportError:
        pass

    candidates = [
        "msedge",
        "chrome",
        "google-chrome",
        "chromium",
        "brave",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for bin_name in candidates:
        resolved = shutil.which(bin_name) or (bin_name if Path(bin_name).exists() else None)
        if resolved:
            subprocess.Popen([resolved, f"--app={url}", "--window-size=1480,920"])
            return False

    webbrowser.open(url)
    return False


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)

    npm_bin = shutil.which("npm")
    if npm_bin and (root / "server.ts").exists() and (root / "node_modules").exists():
        port = find_free_port(3000)
        env = {**os.environ, "PORT": str(port)}
        proc = subprocess.Popen([npm_bin, "run", "dev"], env=env)
    else:
        port = find_free_port(8501)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]
        )

    url = f"http://127.0.0.1:{port}"
    print(f"Launching Fraud Ring Finder PC Workstation on {url} ...")
    if wait_for_server(port):
        blocking_gui = open_native_os_window(url)
        if blocking_gui:
            proc.terminate()
            return

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
