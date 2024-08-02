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
    # q_samples = q_samples[:2]

    random.shuffle(i_samples)
    random.shuffle(q_samples)

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

    # for each i_sample add a field called 'relevance' with value 0
    for i_sample in i_samples:
        i_sample['relevance'] = 0
    for q_sample in q_samples:
        q_sample['relevance'] = 0
        q_sample['completeness'] = 0
        q_sample['question_number'] = -1*roll

    return i_samples, q_samples, roll

def get_user_hash():
    user_data = f"{request.remote_addr}{request.user_agent.string}"
    return hashlib.md5(user_data.encode()).hexdigest()

def get_est_time():
    utc_time = datetime.now(timezone.utc)
    est_time = utc_time - timedelta(hours=4)
    return est_time.strftime('%Y-%B-%d %I:%M%p')

def count_skipped_questions(user_data):
    skipped_questions = 0
    answered_questions = 0
    for i_sample in user_data['i_samples']:
        if i_sample.get('is_info', False):
            continue
        if i_sample['relevance'] == 1:
            skipped_questions += 1
        elif i_sample['relevance'] > 1:
            answered_questions += 1
    for q_sample in user_data['q_samples']:
        if q_sample.get('is_info', False):
            continue
        if q_sample['relevance'] == 1:
            skipped_questions += 1
        elif q_sample['relevance'] > 1:
            answered_questions += 1
        if q_sample['completeness'] == 1:
            skipped_questions += 1
        elif q_sample['completeness'] > 1:
            answered_questions += 1
    return skipped_questions, answered_questions

def get_completion_code(user_data):
    user_hash = get_user_hash()
    skipped_questions, answered_questions = count_skipped_questions(user_data[user_hash])
    answered_code = base36_encode(answered_questions)
    unique_hash_segment = user_hash[-4:]
    pre_checksum_string = answered_code + unique_hash_segment
    checksum = sum(ord(char) for char in pre_checksum_string) % 36
    checksum_character = base36_encode(checksum)
    completion_code = answered_code + unique_hash_segment + checksum_character
    return completion_code

def base36_encode(number):
    assert number >= 0, "Number must be non-negative"
    if number == 0:
        return '0'
    base36 = []
    while number:
        number, i = divmod(number, 36)
        base36.append("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i])
    return ''.join(reversed(base36))

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

    if current_index == total_samples - 1:
        completion_code = get_completion_code(user_data)
        current_set['question1'] = f"Thank you for your participation in this survey!\nYour completion code is: {completion_code}.\nPlease email this code to the researcher (skumar43@gmail.com) to receive your compensation.\nSave a record of this screen for your records."

    is_part_1 = current_index < len(user['i_samples'])

    skipped_questions, answered_questions = count_skipped_questions(user)
    
    # print(f"Current index: {current_index}, Rendering sample_set: {current_set}")
    # print(f"Skipped questions: {skipped_questions}, Answered questions: {answered_questions}")

    return render_template('index.html',
                           sample_set=current_set,
                           set_index=current_index + 1,
                           total_sets=total_samples,
                           is_individual=is_individual,
                           skips = skipped_questions,
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
        relevance = request.form.get('relevance')
        completeness = request.form.get('completeness', 0)  # Default to 0 if not provided
        
        current_sample['relevance'] = int(relevance) if relevance else current_sample.get('relevance', 0)
        if 'completeness' in current_sample:  # This check ensures it's a q_sample
            current_sample['completeness'] = int(completeness) if completeness else current_sample.get('completeness', 0)
        
        append_to_output_file(user_hash, current_sample, {'relevance': relevance, 'completeness': completeness})

    total_samples = len(user['i_samples']) + len(user['q_samples'])
    if 'next' in request.form and current_index < total_samples - 1:
        user['current_set_index'] += 1
    elif 'previous' in request.form and current_index > 0:
        user['current_set_index'] -= 1
    elif 'skip_stage_1' in request.form:
        for i_sample in user['i_samples']:
            i_sample['relevance'] = 1 if i_sample['relevance'] == 0 else i_sample['relevance']
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