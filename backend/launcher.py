import sys
import os

def _get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def _get_bundle_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

# Must run before any project imports
app_dir = _get_app_dir()
bundle_dir = _get_bundle_dir()

os.environ.setdefault("DATA_DIR", os.path.join(app_dir, "data"))
os.environ.setdefault("STATIC_DIR", os.path.join(bundle_dir, "static"))

from dotenv import load_dotenv
load_dotenv(os.path.join(app_dir, ".env"))

import socket, threading, time, webbrowser, multiprocessing

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _wait(port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False

if __name__ == "__main__":
    multiprocessing.freeze_support()
    port = _free_port()
    import uvicorn
    from main import app

    def _serve():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    if _wait(port):
        webbrowser.open(f"http://127.0.0.1:{port}")
    t.join()
