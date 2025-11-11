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
            "current_action_keys": list((_current_action or {}).keys()),
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
            _current_action = {
                "left": 0, "right": 0, "nudge": 0, "plunger": 0,
                "left_f": 0.0, "right_f": 0.0, "nudge_imp": 0.0, "plunger_f": 0.0,
                "duration_s": 0.0
            }
            _last_state = None
            print("[bridge] control: new_ball", file=sys.stderr, flush=True)
            return jsonify({"ok": True, "ack": "new_ball"})
        elif cmd == "end_episode":
            _current_action = {
                "left": 0, "right": 0, "nudge": 0, "plunger": 0,
                "left_f": 0.0, "right_f": 0.0, "nudge_imp": 0.0, "plunger_f": 0.0,
                "duration_s": 0.0
            }
            print("[bridge] control: end_episode", file=sys.stderr, flush=True)
            return jsonify({"ok": True, "ack": "end_episode"})
    return jsonify({"ok": False, "error": "unknown cmd"}), 400

@app.route("/act", methods=["POST"])
def act():
    global _current_action
    data = request.get_json(force=True, silent=True) or {}
    with _lock:
        # Store whatever the env sends; VPX can choose which fields to read.
        _current_action = data
    print(f"[bridge] act: {data}", file=sys.stderr, flush=True)
    return jsonify({"ok": True})

@app.route("/last_state", methods=["GET"])
def last_state():
    with _lock:
        return jsonify(_last_state or {})

@app.route("/state", methods=["POST"])
def state():
    global _last_state, _state_count, _current_action
    data = request.get_json(force=True, silent=True) or {}
    _last_state = {"ts": time.time(), "raw": data}
    _state_count += 1

    act = _current_action or {}

    # Binary outputs (common for VPX VBScript bridges)
    resp = {
        "left": int(act.get("left", 0)),
        "right": int(act.get("right", 0)),
        "nudge": int(act.get("nudge", 0)),
        "plunger": int(act.get("plunger", 0)),   # 🔹 added plunger (binary)
        "duration_s": float(act.get("duration_s", 0.0)),
        "seq": _state_count,
    }

    # Optional continuous fields (safe to include; VPX can ignore them)
    # Useful if you switch the env to continuous controls.
    if "left_f" in act or "right_f" in act or "nudge_imp" in act or "plunger_f" in act:
        resp.update({
            "left_f": float(act.get("left_f", 0.0)),
            "right_f": float(act.get("right_f", 0.0)),
            "nudge_imp": float(act.get("nudge_imp", 0.0)),
            "plunger_f": float(act.get("plunger_f", 0.0)),  # 🔹 added plunger (continuous)
        })

    print(f"the current action is: {_current_action}")
    return jsonify(resp)   # <- critical: jsonify, not an empty return

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
