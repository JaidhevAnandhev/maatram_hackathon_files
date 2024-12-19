from flask import Flask, request, jsonify
import subprocess
import requests

app = Flask(__name__)

def execute_script(text_to_be_inserted):
    command = ['python', '-u', r'c:\xampp\htdocs\aiml_code.py', text_to_be_inserted]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        score_str = result.stdout.strip()[-2:]
        score = int(score_str)
        return score
    except ValueError:
        return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
        return None

@app.route('/api', methods=['POST'])
def handle_post():
    data = request.form
    name = data.get('name')
    response = data.get('response')

    if not name or not response:
        return jsonify({'error': 'Please fill in all fields.'}), 400

    score = execute_script(response)
    if score is not None:
        db_response = requests.post('http://localhost/backend.php', data={'name': name, 'response': response, 'pattern_score': score})
        return db_response.json()
    else:
        return jsonify({'error': 'Invalid response from script.'}), 500

@app.route('/api', methods=['GET'])
def handle_get():
    db_response = requests.get('http://localhost/backend.php')
    return db_response.json()

if __name__ == "__main__":
    app.run(debug=True)
