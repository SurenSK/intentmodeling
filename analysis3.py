
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from collections import OrderedDict, defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
import krippendorff
import pandas as pd
import numpy as np
import json
import time

def mean(l): return sum(l)/len(l)
def variance(lst): return sum((x - sum(lst) / len(lst)) ** 2 for x in lst) / (len(lst) - 1)
def dedup(l): return list(OrderedDict((r[0], r) for r in l).values())
def slope(l): return float(np.polyfit(range(len(l)), l, 1)[0])
def normalize(l, f): return list(np.array(l) - np.array([f * i for i in range(len(l))]))
def getDuration(start_time, end_time): return int((datetime.strptime(end_time, "%Y-%B-%d %I:%M%p") - datetime.strptime(start_time, "%Y-%B-%d %I:%M%p")).total_seconds() / 60)

def kripps(l): return krippendorff.alpha(l, level_of_measurement='ordinal')
def getClusters(data: defaultdict, k: int):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(np.array(list(data.values())))
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_
def sweepKmeans(data: defaultdict, kmin: int, kmax: int): return [(k, getClusters(data, k)[2]) for k in range(kmin, kmax + 1)]
def plotElbow(k_inertias):
    ks = [ki[0] for ki in k_inertias]  # Extract k values
    inertias = [ki[1] for ki in k_inertias]  # Extract inertias

    plt.figure(figsize=(8, 4))
    plt.plot(ks, inertias, 'ko-')
    plt.xticks(ks)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow')
    plt.grid(True)
    plt.show()
from collections import defaultdict
import numpy as np
from scipy.stats import f_oneway

def analyze_clustering(cluster_output, rawDataDict):
    """
    Analyzes clustering by performing ANOVA on the clusters and providing detailed information about each cluster,
    including the number of questions in each cluster.
    
    Parameters:
    - cluster_output (tuple): Output from getClusters containing labels and centroids.
    - rawDataDict (defaultdict): Raw data with QuestionID as keys and lists of ratings as values.
    
    Returns:
    - ANOVA result and detailed cluster information including question counts.
    """
    labels, centroids, _ = cluster_output
    # Prepare data for ANOVA and detailed analysis
    cluster_data = defaultdict(list)
    question_count = defaultdict(int)  # To count questions per cluster

    for question_id, label in zip(rawDataDict.keys(), labels):
        cluster_data[label].extend(rawDataDict[question_id])
        question_count[label] += 1  # Increment question count for the current label
    
    # Extract ratings for each cluster into separate lists and perform ANOVA
    cluster_groups = [group for group in cluster_data.values()]
    anova_result = f_oneway(*cluster_groups)

    # Gathering detailed information about each cluster
    details = {}
    for label, ratings in cluster_data.items():
        details[label] = {
            'question_count': question_count[label],  # Number of questions in the cluster
            'response_count': len(ratings),  # Total number of responses in the cluster
            'mean': np.mean(ratings),
            'std_dev': np.std(ratings)
        }
    
    return anova_result, details
def logClusteringAnalysis(res):
    anova_result, cluster_details = res
    k = len(cluster_details)  # Determine the number of clusters from the dictionary keys
    print(f"k={k}, F-statistic={anova_result.statistic:.2f}, p={anova_result.pvalue:.2e}")
    for cluster_id, details in sorted(cluster_details.items()):
        print(f"\tCluster {cluster_id + 1} - n={details['question_count']}, "
              f"responses={details['response_count']}, "
              f"mean={details['mean']:.2f}, std_dev={details['std_dev']:.2f}")

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

participants = Person.validParticipants

part1_data = []
questions_ratings = defaultdict(list)
questions_summary = defaultdict(None)

for p in participants:
    part1_data.extend([[x[0]]+[x[1]-1] for x in p.dedup_ordered_answers_1 if x[1]!=1])
for resp in part1_data:
    questions_ratings[resp[0]].append(resp[1])
for k in questions_ratings.keys():
    ratings = questions_ratings[k]
    questions_summary[k] = (mean(ratings), variance(ratings))
# plotElbow(sweepKmeans(questions_summary, 1, 10))
# for k in range(2,11):
#     logClusteringAnalysis(analyze_clustering(getClusters(questions_summary,k),questions_ratings))

anova_results = analyze_clustering_2way(getClusters(questions_summary, 3), questions_ratings)
print(anova_results)

print()