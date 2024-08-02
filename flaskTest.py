from flask import Flask, render_template, request, redirect, url_for
import random
import json
import hashlib
from datetime import datetime, timedelta, timezone
from threading import Lock

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a real secret key

# In-memory storage for user-specific data
user_data = {}
input_file_lock = Lock()

def load_and_update_samples(jsonl_file, sample_count, select_lowest=True):
    with input_file_lock:
        with open(jsonl_file, 'r') as file:
            samples = [json.loads(line) for line in file]

        samples_sorted = sorted(samples, key=lambda x: x['count'], reverse=not select_lowest)
        selected_samples = samples_sorted[:sample_count]
        
        for sample in selected_samples:
            sample['count'] += 1

        with open(jsonl_file, 'w') as file:
            for sample in samples:
                file.write(json.dumps(sample) + '\n')

    return selected_samples

def prepare_user_samples():
    i_samples = load_and_update_samples('i_samples.jsonl', 100, select_lowest=True)
    q_samples = load_and_update_samples('q_samples.jsonl', 1, select_lowest=True)
    roll = q_samples[0]['roll']
    i_samples = [sample["data"] for sample in i_samples]
    q_samples = [sample["data"] for sample in q_samples][0]

    random.shuffle(i_samples)

    i_samples.insert(0, {
        'gen#': 'info',
        'prompt#': 'intro1',
        'question': 'Welcome to Part 1 of the survey. In this part, you will be evaluating individual questions. Evaluate each question as if it were from an entirely separate author, without the context of the rest of the survey.',
        'is_info': True
    })
    
    q_samples.insert(0, {
        'gen#': 'info',
        'prompt#': 'intro2',
        'question1': 'Welcome to Part 2 of the survey. In this part, you will be evaluating sets of questions jointly. Evaluate each set of questions as if it were from an entirely separate author, without the context of the rest of the survey.',
        'is_info': True
    })

    q_samples.append({
        'gen#': 'info',
        'prompt#': 'outro',
        'question1': 'Thank you for your participation in this survey!',
        'is_info': True
    })

    return i_samples, q_samples, roll

def get_user_hash():
    user_data = f"{request.remote_addr}{request.user_agent.string}"
    return hashlib.md5(user_data.encode()).hexdigest()

def get_est_time():
    utc_time = datetime.now(timezone.utc)
    est_time = utc_time - timedelta(hours=4)
    return est_time.strftime('%Y-%B-%d %I:%M%p')

output_file_lock = Lock()
def append_to_output_file(user_hash, sample, response):
    current_time = get_est_time()
    
    output = {
        'date-time': current_time,
        'user_hash': user_hash,
        'name': user_data[user_hash]['name'],
        'gen#': sample['gen#'],
        'prompt#': sample['prompt#'],
        'question#': sample.get('question_number', -1),
        'relevance': response['relevance'],
        'completeness': response['completeness']
    }
    
    with output_file_lock:
        with open('survey_responses.jsonl', 'a') as f:
            f.write(json.dumps(output) + '\n')

@app.route('/')
def index():
    user_hash = get_user_hash()
    if user_hash not in user_data:
        i_samples, q_samples, roll = prepare_user_samples()
        user_data[user_hash] = {
            'current_set_index': 0,
            'answers': {},
            'name': '',
            'roll': roll,
            'i_samples': i_samples,
            'q_samples': q_samples
        }

    user = user_data[user_hash]
    current_index = user['current_set_index']
    total_samples = len(user['i_samples']) + len(user['q_samples'])

    if current_index >= total_samples:
        current_index = total_samples - 1
        user['current_set_index'] = current_index

    if current_index < len(user['i_samples']):
        current_set = user['i_samples'][current_index]
        is_individual = True
    else:
        current_set = user['q_samples'][current_index - len(user['i_samples'])]
        is_individual = False

    is_part_1 = current_index < len(user['i_samples'])

    print(f"Current index: {current_index}, Rendering sample_set: {current_set}")

    return render_template('index.html',
                           sample_set=current_set,
                           set_index=current_index + 1,
                           total_sets=total_samples,
                           responses=user['answers'],
                           is_individual=is_individual,
                           i_samples_length=len(user['i_samples']),
                           is_part_1=is_part_1,
                           user_name=user['name'])

@app.route('/update_name', methods=['POST'])
def update_name():
    user_hash = get_user_hash()
    data = request.get_json()
    user_data[user_hash]['name'] = data.get('name', '')
    return '', 204

@app.route('/rate', methods=['POST'])
def rate():
    user_hash = get_user_hash()
    user = user_data[user_hash]
    current_index = user['current_set_index']
    
    current_sample = user['i_samples'][current_index] if current_index < len(user['i_samples']) else user['q_samples'][current_index - len(user['i_samples'])]
    if not current_sample.get('is_info', False):
        current_response = {
            'relevance': request.form.get('relevance'),
            'completeness': request.form.get('completeness')
        }
        user['answers'][str(current_index)] = current_response

        append_to_output_file(user_hash, current_sample, current_response)

    total_samples = len(user['i_samples']) + len(user['q_samples'])
    if 'next' in request.form and current_index < total_samples - 1:
        user['current_set_index'] += 1
    elif 'previous' in request.form and current_index > 0:
        user['current_set_index'] -= 1
    elif 'skip_stage_1' in request.form:
        user['current_set_index'] = len(user['i_samples'])

    user['current_set_index'] = min(user['current_set_index'], total_samples - 1)

    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    user_hash = get_user_hash()
    if user_hash in user_data:
        del user_data[user_hash]
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)