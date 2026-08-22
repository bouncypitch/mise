"""Flask dashboard server. Runs in a background thread while the sim loop
(main thread) writes to shared globals -- single-writer/many-reader, no
locks needed at this polling scale (~300ms)."""
import base64
import json
import os
import threading

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

_lock = threading.Lock()
_shared = {
    "frame_b64": None,
    "state": None,
    "current_action": "Waiting to start...",
    "last_verification": None,
    "status": "INITIALIZING",
}


def update_frame(b64_jpeg):
    with _lock:
        _shared["frame_b64"] = b64_jpeg


def update_state(world_state, current_action=None, last_verification=None, status=None):
    with _lock:
        _shared["state"] = world_state
        if current_action is not None:
            _shared["current_action"] = current_action
        if last_verification is not None:
            _shared["last_verification"] = last_verification
        if status is not None:
            _shared["status"] = status


@app.route("/api/frame")
def api_frame():
    with _lock:
        return jsonify({"frame_b64": _shared["frame_b64"]})


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify({
            "state": _shared["state"],
            "current_action": _shared["current_action"],
            "last_verification": _shared["last_verification"],
            "status": _shared["status"],
        })


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "dashboard.html")


def start_server_in_background(host="127.0.0.1", port=5050):
    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread
