from flask import Flask, Response, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/models', methods=['GET'])
def get_models():
    try:
        env = os.environ.copy()
        env["COLUMNS"] = "10000"  # Use a high column value.
        # Run the command as a shell command.
        command = "/Users/claudiusbirk/.lmstudio/bin/lms ps --json"
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, env=env, shell=True
        )
        return Response(result.stdout, mimetype="application/json")
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": e.stderr.strip()}), 500

@app.route('/unload_all', methods=['POST'])
def unload_all():
    try:
        command = "/Users/claudiusbirk/.lmstudio/bin/lms unload --all"
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, shell=True
        )
        return jsonify({"status": "success", "message": result.stdout.strip()})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": e.stderr.strip()}), 500

if __name__ == '__main__':
    app.run(host='192.168.3.1', port=5051)