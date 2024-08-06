import numpy as np
import pandas as pd
from pingouin import intraclass_corr
from krippendorff import alpha
from scipy.stats import kendalltau
from itertools import combinations

def preprocess_part1(part1_dicts):
    """
    Preprocess Part 1 data into a format suitable for agreement analysis.
    
    :param part1_dicts: List of dictionaries, each representing a participant's responses
    :return: DataFrame with participants as columns and questions as rows
    """
    all_prompts = set()
    for participant_dict in part1_dicts:
        all_prompts.update(participant_dict.keys())
    
    data = {}
    for i, participant_dict in enumerate(part1_dicts):
        for prompt in all_prompts:
            for q in range(5):
                key = f"{prompt}_{q+1}"
                if key not in data:
                    data[key] = [np.nan] * len(part1_dicts)
                if prompt in participant_dict and q < len(participant_dict[prompt]):
                    value = participant_dict[prompt][q]
                    data[key][i] = np.nan if value is None else float(value)
    
    df = pd.DataFrame(data).T
    if df.empty:
        raise ValueError("Preprocessed Part 1 data is empty. Check your input data.")
    return df

def preprocess_part2(part2_dicts):
    """
    Preprocess Part 2 data into a format suitable for agreement analysis.
    
    :param part2_dicts: List of dictionaries, each representing a participant's responses
    :return: Two DataFrames, one for relevance and one for completeness
    """
    all_prompts = set()
    for participant_dict in part2_dicts:
        all_prompts.update(participant_dict.keys())
    
    relevance_data = {}
    completeness_data = {}
    for i, participant_dict in enumerate(part2_dicts):
        for prompt in all_prompts:
            if prompt not in relevance_data:
                relevance_data[prompt] = [np.nan] * len(part2_dicts)
                completeness_data[prompt] = [np.nan] * len(part2_dicts)
            if prompt in participant_dict:
                relevance_data[prompt][i] = float(participant_dict[prompt][0])
                completeness_data[prompt][i] = float(participant_dict[prompt][1])
    
    relevance_df = pd.DataFrame(relevance_data).T
    completeness_df = pd.DataFrame(completeness_data).T
    
    if relevance_df.empty or completeness_df.empty:
        raise ValueError("Preprocessed Part 2 data is empty. Check your input data.")
    
    return relevance_df, completeness_df

def calculate_icc(data):
    """
    Calculate Intraclass Correlation Coefficient (ICC).
    
    :param data: DataFrame with participants as columns and questions as rows
    :return: ICC value or None if calculation is not possible
    """
    try:
        if data.shape[1] < 2:
            print("Insufficient number of raters for ICC calculation.")
            return None
        
        melted_data = data.melt(ignore_index=False).reset_index()
        melted_data.columns = ['targets', 'raters', 'ratings']
        melted_data = melted_data.dropna()
        
        if melted_data.empty:
            print("No valid data for ICC calculation after removing NaN values.")
            return None
        
        icc = intraclass_corr(data=melted_data, targets='targets', raters='raters', ratings='ratings')
        
        if icc.empty:
            print("ICC calculation resulted in empty DataFrame. Check your input data.")
            return None
        
        return icc.loc[icc['Type'] == 'ICC2', 'ICC'].values[0]
    except Exception as e:
        print(f"Error in calculate_icc: {str(e)}")
        print("Data shape:", data.shape)
        print("Data info:")
        print(data.info())
        print("Data head:")
        print(data.head())
        return None

def calculate_krippendorff_alpha(data):
    """
    Calculate Krippendorff's alpha.
    
    :param data: DataFrame with participants as columns and questions as rows
    :return: Krippendorff's alpha value or None if calculation is not possible
    """
    try:
        reliability_data = data.values.T.tolist()
        if not any(reliability_data):
            print("No valid data for Krippendorff's alpha calculation. Check your input data.")
            return None
        return alpha(reliability_data=reliability_data, level_of_measurement='ordinal')
    except Exception as e:
        print(f"Error in calculate_krippendorff_alpha: {str(e)}")
        print("Data shape:", data.shape)
        print("Data info:")
        print(data.info())
        print("Data head:")
        print(data.head())
        return None

def analyze_agreement(part1_dicts, part2_dicts):
    """
    Analyze agreement for both parts of the survey.
    
    :param part1_dicts: List of dictionaries for Part 1 responses
    :param part2_dicts: List of dictionaries for Part 2 responses
    :return: Dictionary containing agreement measures for both parts
    """
    try:
        part1_data = preprocess_part1(part1_dicts)
        part2_relevance, part2_completeness = preprocess_part2(part2_dicts)
        
        results = {
            "Part 1": {
                "ICC": calculate_icc(part1_data),
                "Krippendorff's Alpha": calculate_krippendorff_alpha(part1_data)
            },
            "Part 2 Relevance": {
                "ICC": calculate_icc(part2_relevance),
                "Krippendorff's Alpha": calculate_krippendorff_alpha(part2_relevance)
            },
            "Part 2 Completeness": {
                "ICC": calculate_icc(part2_completeness),
                "Krippendorff's Alpha": calculate_krippendorff_alpha(part2_completeness)
            }
        }
        
        return results
    except Exception as e:
        print(f"Error in analyze_agreement: {str(e)}")
        raise

import matplotlib.pyplot as plt

def plot_trends(list1, list2, list3, title="Trend Lines"):
    """
    Plot three lists of values in separate subplots.
    
    :param list1: List of values for the first subplot (length 100)
    :param list2: List of values for the second subplot (length 70)
    :param list3: List of values for the third subplot (length 70)
    :param title: Title for the entire figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(title)
    
    # Plot first list
    ax1.plot(list1)
    ax1.set_title('List 1 Values')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.set_xlim(0, 100)
    
    # Plot second list
    ax2.plot(list2)
    ax2.set_title('List 2 Values')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_xlim(0, 70)
    
    # Plot third list
    ax3.plot(list3)
    ax3.set_title('List 3 Values')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Value')
    ax3.set_xlim(0, 70)
    
    plt.tight_layout()
    plt.show()

import json
from collections import defaultdict

def load_mixed_data(file_path):
    part1_data = defaultdict(lambda: defaultdict(lambda: [None] * 5))
    part2_data = defaultdict(lambda: defaultdict(tuple))
    part1_ordered = defaultdict(list)
    part2_ordered = defaultdict(lambda: {"relevance": [], "completeness": []})
    
    with open(file_path, 'r') as file:
        lineNum = 0
        parseNum = 0
        for line in file:
            lineNum += 1
            entry = json.loads(line)
            user_hash = entry['user_hash']
            prompt_num = int(entry['prompt#'])
            question_num = int(entry['question#'])
            relevance = int(entry['relevance'])
            completeness = entry['completeness']
            
            if completeness == 0 and 1 <= question_num <= 5:
                # This is a Part 1 entry
                part1_data[user_hash][prompt_num][question_num - 1] = relevance
                part1_ordered[user_hash].append(relevance)
                parseNum += 1
            elif completeness != 0 and question_num < 0:
                # This is a Part 2 entry
                part2_data[user_hash][prompt_num] = (relevance, int(completeness))
                part2_ordered[user_hash]["relevance"].append(relevance)
                part2_ordered[user_hash]["completeness"].append(int(completeness))
                parseNum += 1
    
    print(f"Loaded {parseNum} entries from {lineNum} lines.")
    
    # Convert dict of dicts to list of dicts
    part1_list = [dict(user_data) for user_data in part1_data.values()]
    part2_list = [dict(user_data) for user_data in part2_data.values()]
    
    return part1_list, part2_list, part1_ordered, part2_ordered

# Load mixed data
file_path = 'survey_responses.jsonl'
part1_dicts, part2_dicts, part1_o, part2_o = load_mixed_data(file_path)

# Analyze agreement
results = analyze_agreement(part1_dicts, part2_dicts)
print(results)