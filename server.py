from flask import Flask, request, jsonify, render_template_string
app = Flask(__name__)

@app.route("/ball", methods=["POST"])
@app.route("/ball/", methods=["POST"])
def ball():
    global last_payload
    last_payload = request.get_json(silent=True)
    data = request.get_json(silent=True)
    print("Headers:", dict(request.headers))
    print("JSON:", data)
    return jsonify(ok=True)

@app.get("/")
def index():
    html = """
    <h1>VPX Telemetry</h1>
    <p>POST to <code>/ball</code>. This page is safe to refresh.</p>
    <pre>{{data}}</pre>
    """
    return render_template_string(html, data=last_payload)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
