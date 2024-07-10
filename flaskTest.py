from flask import Flask, render_template, request, redirect, url_for, session
import random
import json
import hashlib
from datetime import datetime, timedelta, timezone

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure random key

def load_samples():
    i_samples = []
    q_samples = []
    
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

i_samples, q_samples = load_samples()

def get_user_hash():
    user_data = f"{request.remote_addr}{request.user_agent.string}"
    return hashlib.md5(user_data.encode()).hexdigest()

@app.route('/')
def index():
    user_hash = get_user_hash()
    if 'user_data' not in session:
        session['user_data'] = {'current_set_index': 0, 'answers': {}}
    current_index = session['user_data']['current_set_index']

    if current_index < len(i_samples):
        current_set = i_samples[current_index]
        return render_template('index.html',
                               sample_set=current_set,
                               set_index=current_index + 1,
                               total_sets=len(i_samples) + len(q_samples),
                               responses=session['user_data']['answers'],
                               is_individual=True,
                               i_samples_length=len(i_samples))
    else:
        current_set = q_samples[current_index - len(i_samples)]
        return render_template('index.html',
                               sample_set=current_set,
                               set_index=current_index + 1,
                               total_sets=len(i_samples) + len(q_samples),
                               responses=session['user_data']['answers'],
                               is_individual=False,
                               i_samples_length=len(i_samples))

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

@app.route('/rate', methods=['POST'])
def rate():
    user_hash = get_user_hash()
    
    if 'user_data' not in session:
        session['user_data'] = {'current_set_index': 0, 'answers': {}}

    current_index = session['user_data']['current_set_index']
    
    # Save current ratings
    current_response = {
        'relevance': request.form.get('relevance'),
        'completeness': request.form.get('completeness')
    }
    session['user_data']['answers'][str(current_index)] = current_response

    # Append the response to the output file
    append_to_output_file(user_hash, current_index, current_response)

    # Navigation logic (Next/Previous/Skip Stage 1)
    total_samples = len(i_samples) + len(q_samples)
    if 'next' in request.form and current_index < total_samples - 1:
        session['user_data']['current_set_index'] += 1
    elif 'previous' in request.form and current_index > 0:
        session['user_data']['current_set_index'] -= 1
    elif 'skip_stage_1' in request.form and current_index < len(i_samples):
        session['user_data']['current_set_index'] = len(i_samples)

    session.modified = True  # Ensure the session is saved
    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    session.clear()  # This will clear all session data
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)