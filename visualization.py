import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

directory = 'group_5_evaluated/'

json_files = [file for file in os.listdir(directory) if file.endswith('.json')]

lengths = []
proximity_scores = []
rouge_scores = []

models = ['llama_7b', 'alpaca_lora_7b', 'bloomz_7b1'] 
colors = ['blue', 'orange', 'green']

for json_file in json_files:
    with open(directory + json_file, 'r') as file:
        data = json.load(file)

    for item in data:
        lengths.append([item['answer_length_output_' + model.replace('-', '_')] for model in models])
        proximity_scores.append([item['proximity_output_' + model.replace('-', '_')] for model in models])
        rouge_scores.append([item['rouge_score_output_' + model.replace('-', '_')] for model in models])

plt.figure(figsize=(15, 5))
for i, model in enumerate(models):
    plt.subplot(1, 3, i + 1)
    sns.violinplot(data=[length[i] for length in lengths], color=colors[i])
    plt.title(f'Answer Lengths - {model}')
    plt.ylabel('Length')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i, model in enumerate(models):
    plt.subplot(1, 3, i + 1)
    sns.violinplot(data=[proximity[i] for proximity in proximity_scores], color=colors[i])
    plt.title(f'Proximity Scores - {model}')
    plt.ylabel('Score')
    plt.xticks(rotation=45)


plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i, model in enumerate(models):
    plt.subplot(1, 3, i + 1)
    sns.violinplot(data=[rouge[i] for rouge in rouge_scores], color=colors[i])
    plt.title(f'ROUGE Scores - {model}')
    plt.ylabel('Score')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

mean_lengths = []
mean_proximity_scores = []
mean_rouge_scores = []

for i in range(len(models)):
    mean_lengths.append(sum(length[i] for length in lengths) / len(lengths))
    mean_proximity_scores.append(sum(proximity[i] for proximity in proximity_scores) / len(proximity_scores))
    mean_rouge_scores.append(sum(rouge[i] for rouge in rouge_scores) / len(rouge_scores))

for i, model in enumerate(models):
    print(f"Model: {model}")
    print(f"Mean Answer Length: {mean_lengths[i]}")
    print(f"Mean Proximity Score: {mean_proximity_scores[i]}")
    print(f"Mean ROUGE Score: {mean_rouge_scores[i]}")
    print()