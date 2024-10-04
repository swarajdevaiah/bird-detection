from flask import Flask, jsonify, render_template, request
import subprocess
import sys
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Get source type and path from the form data
        source_type = request.form.get('source_type')
        source_path = request.form.get('source_path', '').strip()

        if source_type == 'file' and not source_path:
            return jsonify({'message': 'File path is required for file source type'}), 400

        # Call the bird detection script
        result = subprocess.run([sys.executable, 'modify.py', source_type, source_path], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'message': 'Error running script', 'error': result.stderr}), 500

        return jsonify({'message': result.stdout})
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
