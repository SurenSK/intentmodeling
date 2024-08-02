import json

def reorder_questions(q_sample, roll):
    roll = roll % 5
    question_keys = [f'question{i}' for i in range(1, 6)]

    reordered_sample = q_sample.copy()
    for i in range(5):
        old_key = question_keys[i]
        new_key = question_keys[(i - roll) % 5]
        if old_key in q_sample:
            reordered_sample[new_key] = q_sample[old_key]

    return reordered_sample

def genSamples():
    # load samples from ga_output_filtered.jsonl
    i_samples = []
    q_samples = []
    with open('ga_output_filtered.jsonl', 'r') as file:
        for line in file:
            sample = json.loads(line)
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
                            'question_number': i
                        }
                        i_samples.append({"count": 0, "data": new_sample})
    # write i_samples to i_samples.jsonl
    with open('i_samples.jsonl', 'w') as file:
        for sample in i_samples:
            file.write(json.dumps(sample) + '\n')
    with open('q_samples.jsonl', 'w') as file:
        for i in range(5):
            reordered_samples = {"count": 0, "roll": i, "data": [reorder_questions(q_sample, i) for q_sample in q_samples]}
            file.write(json.dumps(reordered_samples) + '\n')
            

def load_and_update_samples(jsonl_file, sample_count, select_lowest=True):
    samples = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            samples.append(json.loads(line))

    samples_sorted = sorted(samples, key=lambda x: x['count'], reverse=not select_lowest)

    selected_samples = samples_sorted[:sample_count]
    for sample in selected_samples:
        sample['count'] += 1

    with open(jsonl_file, 'w') as file:
        for sample in samples:
            file.write(json.dumps(sample) + '\n')

    return selected_samples

genSamples()
print("Done generating samples")
i_samples_selected = load_and_update_samples('i_samples.jsonl', 100, select_lowest=True)
q_samples_selected = load_and_update_samples('q_samples.jsonl', 1, select_lowest=True)

print("Done reloading")
