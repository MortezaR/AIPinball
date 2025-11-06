
from flask import Flask, request, jsonify
from flask_cors import CORS
import time, json, threading, sys

app = Flask(__name__)
CORS(app)

_lock = threading.Lock()
_last_state = None
_current_action = {}
_state_count = 0

def _parse_body():
    # Try JSON normally
    data = request.get_json(force=True, silent=True)
    if isinstance(data, dict):
        return data, 'json'
    # Fallback: raw bytes -> try json loads
    raw = request.data or b''
    try:
        s = raw.decode('utf-8', errors='ignore').strip()
        if s:
            return json.loads(s), 'raw-json'
    except Exception:
        pass
    # Fallback: form fields to dict (in case someone sent x-www-form-urlencoded)
    if request.form:
        try:
            return dict(request.form), 'form'
        except Exception:
            pass
    return {}, 'empty'

@app.route("/health", methods=["GET"])
def health():
    with _lock:
        return jsonify({
            "ok": True,
            "state_posts": _state_count,
            "has_last_state": _last_state is not None,
            "last_state_keys": list((_last_state or {}).get("raw", {}).keys()) if _last_state else [],
        })

@app.route("/debug/clear", methods=["POST"])
def dbg_clear():
    global _last_state, _current_action, _state_count
    with _lock:
        _last_state = None
        _current_action = {}
        _state_count = 0
    return jsonify({"ok": True, "cleared": True})

@app.route("/control", methods=["POST"])
def control():
    data = request.get_json(force=True, silent=True) or {}
    cmd = data.get("cmd")
    global _last_state, _current_action
    with _lock:
        if cmd == "new_ball":
            _current_action = {"left": 0, "right": 0, "nudge": 0, "duration_s": 0.0}
            _last_state = None
            print("[bridge] control: new_ball", file=sys.stderr, flush=True)
            return jsonify({"ok": True, "ack": "new_ball"})
        elif cmd == "end_episode":
            _current_action = {"left": 0, "right": 0, "nudge": 0, "duration_s": 0.0}
            print("[bridge] control: end_episode", file=sys.stderr, flush=True)
            return jsonify({"ok": True, "ack": "end_episode"})
    return jsonify({"ok": False, "error": "unknown cmd"}), 400

@app.route("/act", methods=["POST"])
def act():
    global _current_action
    data = request.get_json(force=True, silent=True) or {}
    with _lock:
        _current_action = data
    print(f"[bridge] act: {data}", file=sys.stderr, flush=True)
    return jsonify({"ok": True})

@app.route("/last_state", methods=["GET"])
def last_state():
    with _lock:
        return jsonify(_last_state or {})

# @app.route("/state", methods=["POST"])
# def state():
#     global _last_state, _state_count
#     data, how = _parse_body()
#     with _lock:
#         _state_count += 1
#         _last_state = {"ts": time.time(), "raw": data}
#     print(f"[bridge] state #{_state_count} via {how}: keys={list(data.keys())}", file=sys.stderr, flush=True)
#     # Return current action for VPX to apply
#     with _lock:
#         act = dict(_current_action) if _current_action else {}
#     return jsonify(act)
@app.route("/state", methods=["POST"])
def state():
    global _last_state, _state_count, _current_action
    data = request.get_json(force=True, silent=True) or {}
    _last_state = {"ts": time.time(), "raw": data}
    _state_count += 1
    act = _current_action or {}
    resp = {
        "left": int(act.get("left", 0)),
        "right": int(act.get("right", 0)),
        "nudge": int(act.get("nudge", 0)),
        "duration_s": float(act.get("duration_s", 0.0)),
        "seq": _state_count,
    }
    print(f"the current action is: {_current_action}")
    return jsonify(resp)   # <- critical: jsonify, not an empty return

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
