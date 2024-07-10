import random
from flask import Flask, render_template, request, redirect, url_for
import json
import hashlib

app = Flask(__name__)
app.secret_key = 'key'

# In-memory storage
responses = {}
i_samples = []
q_samples = []

def get_user_hash():
    user_data = f"{request.remote_addr}{request.user_agent.string}"
    return hashlib.md5(user_data.encode()).hexdigest()

def load_samples():
    global i_samples, q_samples
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

# Call this function at startup
load_samples()

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

from datetime import datetime, timedelta, timezone
def get_est_time():
    utc_time = datetime.now(timezone.utc)
    est_time = utc_time - timedelta(hours=4)
    return est_time.strftime('%Y-%B-%d %I:%M%p')

def append_to_output_file(user_hash, index, response):
    current_time = get_est_time()
    
    if index < len(i_samples):
        sample = i_samples[index]
        question_num = index % 5 + 1  # Calculate question number (1-5)
    else:
        sample = q_samples[index - len(i_samples)]
        question_num = 0  # 0 for question sets
    
    with open('survey_responses.jsonl', 'a') as f:
        output = {
            'date-time': current_time,
            'user_hash': user_hash,
            'gen#': sample['gen#'],
            'prompt#': sample['prompt#'],
            'question#': question_num,
            'relevance': response['relevance'],
            'completeness': response['completeness']
        }
        json.dump(output, f)
        f.write('\n')

def append_to_output_file(user_hash, index, response):
    current_time = get_est_time()
    
    if index < len(i_samples):
        sample = i_samples[index]
        question_num = index % 5 + 1  # Calculate question number (1-5)
    else:
        sample = q_samples[index - len(i_samples)]
        question_num = 0  # 0 for question sets
    
    with open('survey_responses.jsonl', 'a') as f:
        output = {
            'date-time': current_time,
            'user_hash': user_hash,
            'gen#': sample['gen#'],
            'prompt#': sample['prompt#'],
            'question#': question_num,
            'relevance': response['relevance'],
            'completeness': response['completeness']
        }
        json.dump(output, f)
        f.write('\n')

@app.route('/rate', methods=['POST'])
def rate():
    user_hash = get_user_hash()
    
    if user_hash not in responses:
        responses[user_hash] = {'current_set_index': 0, 'answers': {}}

    current_index = responses[user_hash]['current_set_index']
    
    # Save current ratings
    current_response = {
        'relevance': request.form.get('relevance'),
        'completeness': request.form.get('completeness')
    }
    responses[user_hash]['answers'][current_index] = current_response

    # Append the response to the output file
    append_to_output_file(user_hash, current_index, current_response)

    # Navigation logic (Next/Previous)
    total_samples = len(i_samples) + len(q_samples)
    if 'next' in request.form and current_index < total_samples - 1:
        responses[user_hash]['current_set_index'] += 1
    elif 'previous' in request.form and current_index > 0:
        responses[user_hash]['current_set_index'] -= 1

    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True)