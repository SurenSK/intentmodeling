from flask import Flask, render_template, request, redirect, url_for, session
import json
import os
import hashlib

app = Flask(__name__)
app.secret_key = 'key'
# 

def load_samples():
    samples = []
    try:
        with open('ga_output.jsonl', 'r') as file:
            for line in file:
                samples.append(json.loads(line))
    except FileNotFoundError:
        print("ga_output.jsonl file not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSONL: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    return samples

def load_responses():
    if os.path.exists('survey_responses.json'):
        with open('survey_responses.json', 'r') as f:
            return json.load(f)
    return {}

def save_responses(responses):
    with open('survey_responses.json', 'w') as f:
        json.dump(responses, f)

def get_user_hash():
    user_data = f"{request.remote_addr}{request.user_agent.string}"
    return hashlib.md5(user_data.encode()).hexdigest()

samples = load_samples()
responses = load_responses()

@app.route('/')
def index():
    user_hash = get_user_hash()
    if user_hash not in responses:
        responses[user_hash] = {'current_set_index': 0, 'answers': {}}
    
    current_set_index = responses[user_hash]['current_set_index']
    
    if samples:
        current_set = samples[current_set_index]
        return render_template('index.html', 
                               sample_set=current_set, 
                               set_index=current_set_index + 1, 
                               total_sets=len(samples),
                               responses=responses[user_hash]['answers'])
    else:
        return "Error loading samples", 500

@app.route('/rate', methods=['POST'])
def rate():
    user_hash = get_user_hash()
    current_index = responses[user_hash]['current_set_index']
    
    # Save current ratings
    responses[user_hash]['answers'][current_index] = {
        'relevance': request.form.get('relevance'),
        'completeness': request.form.get('completeness')
    }

    # Navigation logic (Next/Previous)
    if 'next' in request.form and current_index < len(samples) - 1:
        responses[user_hash]['current_set_index'] += 1
    elif 'previous' in request.form and current_index > 0:
        responses[user_hash]['current_set_index'] -= 1

    # Save responses to a file
    save_responses(responses)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)