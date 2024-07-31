from flask import Flask, render_template, request, redirect, url_for, session
import random
import json
import hashlib
from datetime import datetime, timedelta, timezone
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

def load_samples():
    i_samples = []
    q_samples = []
    sample_counts = defaultdict(int)
    
    try:
        with open('ga_output_filtered.jsonl', 'r') as file:
            for line in file:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

                if sample.get('survey', False) or True:
                    q_samples.append(sample)
                    for i in range(1, 6):
                        question_key = f'question{i}'
                        if question_key in sample:
                            new_sample = {
                                'gen#': sample['gen#'],
                                'prompt#': sample['prompt#'],
                                'score': sample.get('score',0),
                                'valid': sample.get('valid', True),
                                'prompt': sample.get('prompt', ""),
                                'question': sample[question_key],
                                'relevance': 0,
                                'question_number': i
                            }
                            i_samples.append(new_sample)

        try:
            with open('sample_counts.json', 'r') as count_file:
                sample_counts = json.load(count_file)
        except FileNotFoundError:
            print("sample_counts.json not found, starting with empty counts")

    except FileNotFoundError:
        print("ga_output.jsonl file not found")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    i_samples.sort(key=lambda x: sample_counts.get(f"{x['gen#']}_{x['prompt#']}_{x['question_number']}", 0))
    i_samples = i_samples[:100]
    random.shuffle(i_samples)
    # only load 5 q_samples for now
    q_samples = q_samples[:5]
    random.shuffle(q_samples)
    for sample in q_samples:
        sample['relevance'] = 0
        sample['completeness'] = 0

    i_samples.insert(0, {
        'gen#': 'info',
        'prompt#': 'intro1',
        'question': 'Welcome to Part 1 of the survey. In this part, you will be evaluating individual questions. Evaluate each question as if it were from an entirely separate author, without the context of the rest of the survey.',
        'is_info': True
    })
    
    q_samples.insert(0, {
        'gen#': 'info',
        'prompt#': 'intro2',
        'question': 'Welcome to Part 2 of the survey. In this part, you will be evaluating sets of questions jointly. Evaluate each set of questions as if it were from an entirely separate author, without the context of the rest of the survey.',
        'is_info': True
    })

    q_samples.append({
        'gen#': 'info',
        'prompt#': 'outro',
        'question': 'Thank you for your participation in this survey!',
        'is_info': True
    })

    return i_samples, q_samples, sample_counts

i_samples, q_samples, sample_counts = load_samples()

import string
import random
def base36encode(number):
    alphabet = string.digits + string.ascii_uppercase
    if number == 0:
        return alphabet[0]
    base36 = []
    while number:
        number, i = divmod(number, 36)
        base36.append(alphabet[i])
    return ''.join(reversed(base36))

def generate_completion_code(questions_answered, user_hash):
    # Convert questions_answered to base 36 (0-9 + A-Z)
    questions_part = base36encode(questions_answered).zfill(2).upper()
    
    # Get last 2 characters of user_hash (uppercase)
    hash_part = user_hash[-2:].upper()
    
    # Combine parts
    code = questions_part + hash_part
    
    # Generate checksum (sum of ASCII values mod 36)
    checksum = sum(ord(c) for c in code) % 36
    checksum_char = base36encode(checksum)
    
    return code + checksum_char

def get_user_hash():
    user_data = f"{request.remote_addr}{request.user_agent.string}"
    return hashlib.md5(user_data.encode()).hexdigest()

def get_est_time():
    utc_time = datetime.now(timezone.utc)
    est_time = utc_time - timedelta(hours=4)
    return est_time.strftime('%Y-%B-%d %I:%M%p')

def append_to_output_file(user_hash, index, response):
    current_time = get_est_time()
    
    if index < len(i_samples):
        sample = i_samples[index]
        question_num = sample['question_number']
    else:
        q_index = index - len(i_samples)
        if q_index >= len(q_samples):
            print(f"Warning: Attempting to access q_samples out of range. Index: {q_index}, q_samples length: {len(q_samples)}")
            return
        sample = q_samples[q_index]
        question_num = 0  # 0 for question sets
    
    with open('survey_responses.jsonl', 'a') as f:
        output = {
            'date-time': current_time,
            'user_hash': user_hash,
            'name': session['user_data'].get('name', ''),  # Add this line
            'gen#': sample['gen#'],
            'prompt#': sample['prompt#'],
            'question#': question_num,
            'relevance': response['relevance'],
            'completeness': response['completeness']
        }
        json.dump(output, f)
        f.write('\n')

    sample_key = f"{sample['gen#']}_{sample['prompt#']}_{question_num}"
    sample_counts[sample_key] = sample_counts.get(sample_key, 0) + 1

    with open('sample_counts.json', 'w') as count_file:
        json.dump(sample_counts, count_file)


@app.route('/')
def index():
    user_hash = get_user_hash()
    if 'user_data' not in session:
        session['user_data'] = {'current_set_index': 0, 'answers': {}, 'name': ''}
    current_index = session['user_data']['current_set_index']

    total_samples = len(i_samples) + len(q_samples)
    if current_index >= total_samples:
        current_index = total_samples - 1
        session['user_data']['current_set_index'] = current_index

    if current_index < len(i_samples):
        current_set = i_samples[current_index]
        is_individual = True
    else:
        current_set = q_samples[current_index - len(i_samples)]
        is_individual = False
    
    is_part_1 = current_index < len(i_samples)
    
    # Initialize counters

    skipped_questions = 0
    total_questions = 0
    questions_answered = 0
    
    # Count skipped and total questions
    for sample in i_samples:
        if sample.get('gen#') == 'info':
            continue
        total_questions += 1
        try:
            skipped_questions += sample["relevance"]==1
            questions_answered += sample["relevance"]>1
        except:
            pass
    for sample in q_samples:
        if sample.get('gen#') == 'info':
            continue
        total_questions += 1
        try:
            skipped_questions += sample["relevance"]==1
            skipped_questions += sample["completeness"]==1
            questions_answered += sample["relevance"]>1
            questions_answered += sample["completeness"]>1
        except:
            pass
        

    # Handle outro screen
    if current_set.get('prompt#') == 'outro':
        completion_code = generate_completion_code(questions_answered, user_hash)
        current_set['question'] = current_set.get('question', '') + f"\nYour completion code is: {completion_code}\nPlease send this code to the researcher to receive your compensation. Save a screenshot of this page for your records."

    # Handle info screens
    if current_set.get('is_info', False):
        current_set['question'] = current_set.get('question', "Welcome to the survey.")

    completion_code = generate_completion_code(questions_answered, user_hash)
    return render_template('index.html',
                           sample_set=current_set,
                           set_index=current_index + 1,
                           total_sets=total_samples,
                           responses=session['user_data']['answers'],
                           is_individual=is_individual,
                           i_samples_length=len(i_samples),
                           is_part_1=is_part_1,
                           completion_code=completion_code,
                           user_name=session['user_data'].get('name', ''))


@app.route('/update_name', methods=['POST'])
def update_name():
    data = request.get_json()
    session['user_data']['name'] = data.get('name', '')
    session.modified = True
    return '', 204  # Return an empty response with a 204 No Content status

@app.route('/rate', methods=['POST'])
def rate():
    user_hash = get_user_hash()
    
    if 'user_data' not in session:
        session['user_data'] = {'current_set_index': 0, 'answers': {}, 'name': ''}

    current_index = session['user_data']['current_set_index']
    
    if current_index < len(i_samples):
        current_sample = i_samples[current_index]
    else:
        current_sample = q_samples[current_index - len(i_samples)]

    if not current_sample.get('is_info', False):
        relevance = request.form.get('relevance')
        completeness = request.form.get('completeness')
        
        current_sample['relevance'] = int(relevance) if relevance else 1
        current_sample['completeness'] = int(completeness) if completeness else 1

        session['user_data']['answers'][str(current_index)] = {
            'relevance': current_sample['relevance'],
            'completeness': current_sample['completeness']
        }

        append_to_output_file(user_hash, current_index, session['user_data']['answers'][str(current_index)])

    total_samples = len(i_samples) + len(q_samples)
    if 'next' in request.form and current_index < total_samples - 1:
        session['user_data']['current_set_index'] += 1
    elif 'previous' in request.form and current_index > 0:
        session['user_data']['current_set_index'] -= 1
    elif 'skip_stage_1' in request.form:
        session['user_data']['current_set_index'] = len(i_samples)

    session['user_data']['current_set_index'] = min(session['user_data']['current_set_index'], total_samples - 1)

    session.modified = True
    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    session.clear()
    for sample in i_samples:
        sample['relevance'] = 0
    for sample in q_samples:
        sample['relevance'] = 0
        sample['completeness'] = 0
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)