import os
import spacy
from rouge_score import rouge_scorer
import json
import numpy as np
from scipy.spatial.distance import cosine

nlp = spacy.load("en_core_web_md")

def calculate_answer_length(answer):
    return len(answer.split())

# Calculate proximity between question and answer 
def calculate_proximity(question, answer):
    question_vector = nlp(question).vector
    answer_vector = nlp(answer).vector if answer.strip() else nlp('').vector
    
    if np.count_nonzero(question_vector) <= 0 or np.count_nonzero(answer_vector) <= 0:
        return 0.0

    proximity_score = 1 - cosine(question_vector, answer_vector)
    proximity_score = np.clip(proximity_score, 0.0, 1.0)

    return proximity_score

# Calculate ROUGE-1 F1 score
def calculate_rouge_score(answer, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(answer, reference)
    return scores['rouge1'].fmeasure

def create_subset(data_points):
    processed_dataset = []

    for data_point in data_points:
        if isinstance(data_point, dict) and all(key in data_point for key in required_keys):
            question_squad = data_point["question_squad"]
            answer_squad = data_point["answer_squad"]
            text = data_point["text"]

            subset_data_point = {
                "question_squad": question_squad,
                "answer_squad": answer_squad,
                "text": text,
            }

            for model_key in ["output_llama-7b", "output_alpaca-lora-7b", "output_bloomz-7b1"]:
                answer_key = f"answer_{model_key.replace('-', '_')}"
                answer_length_key = f"answer_length_{model_key.replace('-', '_')}"
                proximity_key = f"proximity_{model_key.replace('-', '_')}"
                rouge_score_key = f"rouge_score_{model_key.replace('-', '_')}"

                answer = data_point[model_key]
                answer_length = calculate_answer_length(answer)
                proximity_score = calculate_proximity(question_squad, answer)
                rouge_score = calculate_rouge_score(answer, answer_squad)

                subset_data_point[answer_key] = answer
                subset_data_point[answer_length_key] = answer_length
                subset_data_point[proximity_key] = proximity_score
                subset_data_point[rouge_score_key] = rouge_score

            processed_dataset.append(subset_data_point)

    return processed_dataset

def read_dataset_file(file_path):
    dataset = []

    with open(file_path, "r") as file:
        try:
            data = json.load(file)
            if isinstance(data, list):
                dataset.extend(data)
            else:
                dataset.append(data)
        except json.JSONDecodeError:
            print("Invalid JSON data:", file_path)

    return create_subset(dataset)

def process_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_name = file_name.replace("cleaned", "evaluated")
        output_file_path = os.path.join(output_folder, output_file_name)
        subset = read_dataset_file(input_file_path)

        with open(output_file_path, 'w') as f:
            json.dump(subset, f, indent=2)

input_folder = "group_5_cleaned"
output_folder = "group_5_evaluated"


required_keys = ["question_squad","answer_squad", "output_llama-7b", "output_alpaca-lora-7b", "output_bloomz-7b1", "text"]

process_files(input_folder, output_folder)
