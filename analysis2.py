from collections import OrderedDict
from datetime import datetime
import krippendorff
import numpy as np
import json
import time

def mean(l): return sum(l)/len(l)
def dedup(l): return list(OrderedDict((r[0], r) for r in l).values())
def slope(l): return float(np.polyfit(range(len(l)), l, 1)[0])
def normalize(l, f): return list(np.array(l) - np.array([f * i for i in range(len(l))]))
def getDuration(start_time, end_time): return int((datetime.strptime(end_time, "%Y-%B-%d %I:%M%p") - datetime.strptime(start_time, "%Y-%B-%d %I:%M%p")).total_seconds() / 60)

def kripps(l): return krippendorff.alpha(l, level_of_measurement='ordinal')

# outlierHashes = ["2f0892ec1836e6f1d793e8271480a68b", "b91e2ea447b14f58c9c21b7b7aa31755"]

outlierHashes = []
outA = "2f0892ec1836e6f1d793e8271480a68b"
outB = "b91e2ea447b14f58c9c21b7b7aa31755"
cUser = None
class Person:
    validParticipants = []
    def __init__(self, userID):
        self.user_hash = userID
        self.time_start = None
        self.time_end = None
        self.ordered_answers_1=[] # lists like [gen#_prompt#_question#,relevance,completeness] ex [1_46_2,6,0]
        self.ordered_answers_2=[] # lists like [gen#_prompt#_question#,relevance,completeness] ex [1_23_0,4,3]

    def update(self, t, gen, qset, qnum, rel, com):
        if self.time_start is None:
            self.time_start = t
        self.time_end = t
        if qnum<1:
            qnum=0
            self.ordered_answers_2.append([f"{gen}_{qset}_{qnum}", rel, com])
        else:
            self.ordered_answers_1.append([f"{gen}_{qset}_{qnum}", rel, com])

    def process(self):
        cUser = self
        self.quiz_duration = getDuration(self.time_start, self.time_end)
        self.dedup_ordered_answers_1 = dedup(self.ordered_answers_1)
        self.dedup_ordered_answers_2 = dedup(self.ordered_answers_2)
        self.part1_slope = slope([x[1] for x in self.ordered_answers_1]) if len(self.dedup_ordered_answers_1)>1 else None
        self.part2_r_slope = slope([x[1] for x in self.ordered_answers_2]) if len(self.dedup_ordered_answers_2)>1 else None
        self.part2_c_slope = slope([x[2] for x in self.ordered_answers_2]) if len(self.dedup_ordered_answers_2)>1 else None
        self.part1_normalized_r = normalize([x[1] for x in self.dedup_ordered_answers_1], self.part1_slope) if self.part1_slope else None
        self.part2_normalized_r = normalize([x[1] for x in self.dedup_ordered_answers_2], self.part2_r_slope) if self.part2_r_slope else None
        self.part2_normalized_c = normalize([x[2] for x in self.dedup_ordered_answers_2], self.part2_c_slope) if self.part2_c_slope else None
        self.valid = self.part2_normalized_r and len(self.part2_normalized_r)>65 and self.user_hash not in outlierHashes
        if self.valid:
            Person.validParticipants.append(self)
            self.keys_1 = [x[0] for x in self.dedup_ordered_answers_1]
            temp = list(zip(self.keys_1, self.part1_normalized_r))

            # gens_r = {}
            # for k,r in temp:
            #     k_=k.split("_")[0]
            #     if k_ not in gens_r:
            #         gens_r[k_]=[]
            #     gens_r[k_].append(r)
            # self.gens_r_avg = {}
            # for gen in gens_r:
            #     self.gens_r_avg[gen]=mean(gens_r[gen])
            # print(self.gens_r_avg)

            self.keys_2 = [x[0] for x in self.dedup_ordered_answers_2]
            temp = list(zip(self.keys_2, self.part2_normalized_r))
            temp.sort()
            self.part2_final_r = [x[1] for x in temp]
            control = mean(self.part2_final_r[-10:])
            self.part2_final_r = [x-control for x in self.part2_final_r]

            gens_r = {}
            for k,r in zip(self.keys_2, self.part2_final_r):
                k_=k.split("_")[0]
                if k_ not in gens_r:
                    gens_r[k_]=[]
                gens_r[k_].append(r)
            gens_r_avg = {}
            for gen in gens_r.keys():
                gens_r_avg[gen]=mean(gens_r[gen])
            orderType = []
            orderType.append("S")
            orderType.append("1" if gens_r_avg["4"]>gens_r_avg["0"] else "0")
            orderType.append("1" if gens_r_avg["1"]>gens_r_avg["0"] or gens_r_avg["2"]>gens_r_avg["0"] else "0")
            orderType.append("E")
            self.orderType = "".join(orderType)
            self.genScores = []
            self.genScores.append(gens_r_avg["0"])
            self.genScores.append(gens_r_avg["1"])
            self.genScores.append(gens_r_avg["2"])
            self.genScores.append(gens_r_avg["4"])

            temp = list(zip(self.keys_2, self.part2_normalized_c))
            temp.sort()
            self.part2_final_c = [x[1] for x in temp]
            control = mean(self.part2_final_c[-10:])
            self.rc_corr = np.corrcoef(self.part2_final_r, self.part2_final_c)[0, 1]
            self.part2_final_c = [x-control for x in self.part2_final_c]

participants = {}
def load_data(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            date = entry['date-time']
            user_hash = entry['user_hash']
            prompt_num = int(entry['prompt#'])
            question_num = int(entry['question#'])
            relevance = int(entry['relevance'])
            completeness = int(entry['completeness'])
            gen_num = int(entry['gen#'])
            if user_hash == "c1c476a501fb613dcd32ff7a55cdcae4" or user_hash=="a930044b0bcc33fc743ab236ebdf3c52":
                dateKey = date.split(" ")[0]
                user_hash += dateKey[-2:]
            if user_hash not in participants:
                participants[user_hash] = Person(user_hash)
            participants[user_hash].update(date, gen_num, prompt_num, question_num, relevance, completeness)

t0 = time.time()
load_data("survey_responses.jsonl")
print(f"+{time.time()-t0}s\tLoaded Data")

t0 = time.time()
[p.process() for p in participants.values()]
print(f"+{time.time()-t0}s\tDeDuped+Normalized Data")

for p in Person.validParticipants:
    print(f"User Hash: {p.user_hash[-4:]}, Type: {p.orderType} Quiz Duration: {p.quiz_duration:4d}, R-C Corr: {p.rc_corr:0.2f}")

data_r = {}
for p in Person.validParticipants:
    # if p.user_hash == "4e6df1eb08860c87d2a436518d2ba66f":
    #     print("Skipped ba66f outlier")
    #     continue
    if p.orderType not in data_r:
        data_r[p.orderType]=[]
    data_r[p.orderType].append(p.part2_final_r)

for k in data_r.keys():
    print(k, len(data_r[k]), kripps(data_r[k]))

gens_r = {}
for p in Person.validParticipants:
    for answer in p.dedup_ordered_answers_2:
        gen, _, _ = answer[0].split("_")  # Split the identifier to extract the generation number
        if gen not in gens_r:
            gens_r[gen] = []
        gens_r[gen].append(answer[1])  # Append relevance rating

# Check the data size for sanity check
for gen, scores in gens_r.items():
    print(f"Generation {gen} has {len(scores)} data points.")
print(f"Total {len(Person.validParticipants)} data points")


import scipy.stats as stats
score_lists = [scores for scores in gens_r.values()]
anova_result = stats.f_oneway(*score_lists)
print(f"ANOVA F-statistic: {anova_result.statistic}, P-value: {anova_result.pvalue}")


data_r = []
for p in Person.validParticipants:
    data_r.append([p.user_hash]+[p.orderType]+p.part2_final_r)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Converting to DataFrame
df = pd.DataFrame(data_r, columns=['user_hash', 'orderType'] + [f'r_{i}' for i in range(70)])

# Extracting the last 70 values for PCA
pca_data = df.iloc[:, -70:]  # Last 70 columns

# Applying PCA to reduce dimensions to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_data)

# Adding the PCA results back to the dataframe
df['pca_one'] = pca_result[:, 0]
df['pca_two'] = pca_result[:, 1]

# Plotting the data
plt.figure(figsize=(10, 6))
color_map = {
    'S00E': '#0033CC',  # Dark Blue
    'S10E': '#6699FF',  # Light Blue
    'S11E': '#CC0000',  # Dark Red
    'S01E': '#FF6666'   # Light Red
}

colors = df['orderType'].map(color_map)
scatter = plt.scatter(df['pca_one'], df['pca_two'], c=colors, alpha=0.5)
labels = list(color_map.keys())
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10) for label in labels]
plt.legend(handles, labels, title="Order Type", loc='upper left', bbox_to_anchor=(1, 1))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Survey Responses')
plt.grid(True)
# Using mplcursors to show user_hash on hover
import mplcursors
cursor = mplcursors.cursor(scatter, hover=True)
@cursor.connect("add")
def on_add(sel):
    sel.annotation.set_text(df['user_hash'].iloc[sel.index])
    print(df['user_hash'].iloc[sel.index])
plt.show()

import csv
filename = 'data_r.csv'

# Writing to the file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in data_r:
        writer.writerow(row)

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
participantTypes = ["S00E", "S01E", "S10E", "S11E"]
from itertools import combinations

def all_combinations(lst):
    combo_list = []
    # Loop through all possible lengths of combinations
    for r in range(1, len(lst) + 1):
        # Generate combinations of length r
        combo_list.extend(combinations(lst, r))
    return [list(item) for item in combo_list]
thsdSlices = all_combinations(participantTypes)
thsdSlices = [['S01E', 'S11E'], ['S00E', 'S10E'], ['S01E', 'S11E', 'S00E', 'S10E']]

for ptypes in thsdSlices:
    # First, gather all relevant data into a list of dictionaries
    data_list = []
    numP = 0
    for participant in Person.validParticipants:
        if participant.orderType not in ptypes:
            continue
        numP += 1
        for score, key in zip(participant.part2_final_r, participant.keys_2):
            gen = key.split("_")[0]  # Extracting the generation number from the identifier
            data_list.append({'Scores': score, 'Generation': gen})
    print(ptypes, numP)

    # Convert the list of dictionaries into a DataFrame
    data = pd.DataFrame(data_list)

    # Ensure data types are correct, especially if 'Generation' should be treated as categorical
    data['Scores'] = pd.to_numeric(data['Scores'], errors='coerce')
    data['Generation'] = data['Generation'].astype('category')

    # Filter out the generations not part of your hypothesis if necessary
    data = data[data['Generation'].isin(['0', '1', '2'])]

    # Perform Tukey's HSD test
    tukey = pairwise_tukeyhsd(endog=data['Scores'], groups=data['Generation'], alpha=0.05)

    # Print the results
    print(tukey)

    # Optionally, print a summary table for specific comparisons involving Generation 0
    print("Specific Comparisons for Generation 0:")
    results = tukey.summary().data[1:]  # Skip the header row

data = []
classes = ["Anti-Optimization (17)", "Pro-Optimization (31)"]
classes = ["Uncorrelated RC (15)", "Correlated RC (33)"]
for p in Person.validParticipants:
    agreement = classes[0] if p.orderType == "S10E" or p.orderType == "S00E" else classes[1]
    agreement = classes[0] if p.rc_corr>0.8 else classes[1]
    data.extend([[agreement] + ["_".join(item[0].split("_")[:2])] + [item[1]] for item in p.dedup_ordered_answers_1])
# Extract x, y, and labels from the data
data = sorted(data, key=lambda x: int(x[1].split('_')[0]), reverse=True)
data = [[item[0], item[1], item[2] - 1] for item in data if item[2] != 1]

# Convert the data list to a DataFrame
df = pd.DataFrame(data, columns=['Agreement', 'Question', 'Rating'])

# Prepare the data for a boxplot
df_sorted = df.sort_values(by='Question')
# Extract generation number from the 'Question' field
df_sorted['Generation'] = df_sorted['Question'].apply(lambda x: x.split('_')[0])

# Group by Agreement and Generation and calculate the average rating
average_ratings = df_sorted.groupby(['Agreement', 'Generation']).mean().reset_index()

import seaborn as sns
# Creating a boxplot
plt.figure(figsize=(12, 8))
boxplot = sns.boxplot(x='Question', y='Rating', hue='Agreement', data=df_sorted, palette=['#ff9999', '#99ff99'])
plt.xticks(rotation=90)
plt.title('Survey Ratings by Question and RC Correlation')
plt.xlabel('Question')
plt.ylabel('Rating')
plt.legend(title='Cluster', loc='upper left', bbox_to_anchor=(1, 1))

cluster_averages = [4.5, 3.2, 4.8, 3.7]  # Placeholder values
other_cluster_averages = [4.1, 3.5, 4.2, 3.8]  # Placeholder values
cluster_averages = average_ratings[average_ratings['Agreement'] == classes[0]]['Rating'].tolist()
other_cluster_averages = average_ratings[average_ratings['Agreement'] == classes[1]]['Rating'].tolist()

colors = ['#ff9999', '#99ff99']  # Colors used in the boxplot
partition_indices = [19, 38, 60, 70]
for i, avg in enumerate(cluster_averages):
    start_idx = 0 if i == 0 else partition_indices[i-1]
    end_idx = partition_indices[i]
    plt.hlines(y=avg, xmin=start_idx - 0.5, xmax=end_idx - 0.5, colors='black', linestyles='-', linewidth=3)
    plt.hlines(y=avg, xmin=start_idx - 0.5, xmax=end_idx - 0.5, colors=colors[0], linestyles='-', linewidth=2)

for i, avg in enumerate(other_cluster_averages):
    start_idx = 0 if i == 0 else partition_indices[i-1]
    end_idx = partition_indices[i]
    plt.hlines(y=avg, xmin=start_idx - 0.5, xmax=end_idx - 0.5, colors='black', linestyles='-', linewidth=3)
    plt.hlines(y=avg, xmin=start_idx - 0.5, xmax=end_idx - 0.5, colors=colors[1], linestyles='-', linewidth=2)

partition_indices = [19, 38, 60]
# Manually add the partition lines (insert indices in the list below)
for idx in partition_indices:  # Update these indices to match your data
    plt.axvline(x=idx - 0.5, color='black', linewidth=2, linestyle='-')
plt.subplots_adjust(left=0.05, right=0.85)
plt.xlim(left=-0.5, right=len(df_sorted['Question'].unique()) - 0.5)
plt.show()

a=participants["c1c476a501fb613dcd32ff7a55cdcae412"]
print("Done")