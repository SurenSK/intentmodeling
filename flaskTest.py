from flask import Flask, render_template, request, redirect, url_for, session
import json
import os
import hashlib
import random

app = Flask(__name__)
app.secret_key = 'key'
def load_samples():
    i_samples = []
    q_samples = []
    try:
        with open('ga_output.jsonl', 'r') as file:
            for line in file:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

                if sample.get('survey', False):
                    q_samples.append(sample)
                    for i in range(1, 6):
                        question_key = f'question{i}'
                        if question_key in sample:
                            new_sample = {
                                'gen#': sample['gen#'],
                                'prompt#': sample['prompt#'],
                                'score': sample['score'],
                                'valid': sample['valid'],
                                'prompt': sample['prompt'],
                                'question': sample[question_key]
                            }
                            i_samples.append(new_sample)

    except FileNotFoundError:
        print("ga_output.jsonl file not found")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    random.shuffle(i_samples)
    random.shuffle(q_samples)
    return i_samples, q_samples

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

i_samples, q_samples = load_samples()
responses = load_responses()
@app.route('/')
def index():
    user_hash = get_user_hash()
    if user_hash not in responses:
        responses[user_hash] = {'current_set_index': 0, 'answers': {}}

    current_index = responses[user_hash]['current_set_index']

    if current_index < len(i_samples):
        current_set = i_samples[current_index]
        return render_template('index.html',
                               sample_set=current_set,
                               set_index=current_index + 1,
                               total_sets=len(i_samples) + len(q_samples),
                               responses=responses[user_hash]['answers'],
                               is_individual=True)
    else:
        current_set = q_samples[current_index - len(i_samples)]
        return render_template('index.html',
                               sample_set=current_set,
                               set_index=current_index + 1,
                               total_sets=len(i_samples) + len(q_samples),
                               responses=responses[user_hash]['answers'],
                               is_individual=False)
    
@app.route('/rate', methods=['POST'])
@app.route('/rate', methods=['POST'])
def rate():
    user_hash = get_user_hash()
    
    # Ensure user data is initialized
    if user_hash not in responses:
        responses[user_hash] = {'current_set_index': 0, 'answers': {}}

    current_index = responses[user_hash]['current_set_index']
    
    # Save current ratings
    responses[user_hash]['answers'][current_index] = {
        'relevance': request.form.get('relevance'),
        'completeness': request.form.get('completeness')
    }

    # Navigation logic (Next/Previous)
    total_samples = len(i_samples) + len(q_samples)
    if 'next' in request.form and current_index < total_samples - 1:
        responses[user_hash]['current_set_index'] += 1
    elif 'previous' in request.form and current_index > 0:
        responses[user_hash]['current_set_index'] -= 1

    # Save responses to a file
    save_responses(responses)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)