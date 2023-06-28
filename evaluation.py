import os
import spacy
from rouge_score import rouge_scorer
import json

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Calculate answer length
def calculate_answer_length(answer):
    return len(answer.split())

# Calculate proximity between question and answer in context
def calculate_proximity(question, answer, context):
    doc = nlp(context)
    question_similarity = nlp(question).similarity(doc)
    
    if answer.strip():
        answer_similarity = nlp(answer).similarity(doc)
    else:
        answer_similarity = 0.0

    # Calculate the average proximity score
    proximity_score = (question_similarity + answer_similarity) / 2

    # Return the proximity score
    return proximity_score

def calculate_rouge_score(answer, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(answer, reference)

    # Return the ROUGE-1 F1 score
    return scores['rouge1'].fmeasure

# Apply selection rules and create a subset
def create_subset(data_points):
    processed_dataset = []
    for data_point in data_points:
        if isinstance(data_point, dict):
            question_squad = data_point.get("question_squad")
            answer_output_llama_7b = data_point.get("output_llama-7b")
            answer_output_alpaca_lora_7b = data_point.get("output_alpaca-lora-7b")
            answer_output_bloomz_7b1 = data_point.get("output_bloomz-7b1")
            text = data_point.get("text")

            if question_squad and answer_output_llama_7b and answer_output_alpaca_lora_7b and answer_output_bloomz_7b1 and text:
                answer_length_output_llama_7b = calculate_answer_length(answer_output_llama_7b)
                proximity_output_llama_7b = calculate_proximity(question_squad, answer_output_llama_7b, text)
                rouge_score_output_llama_7b = calculate_rouge_score(answer_output_llama_7b, text)

                answer_length_output_alpaca_lora_7b = calculate_answer_length(answer_output_alpaca_lora_7b)
                proximity_output_alpaca_lora_7b = calculate_proximity(question_squad, answer_output_alpaca_lora_7b, text)
                rouge_score_output_alpaca_lora_7b = calculate_rouge_score(answer_output_alpaca_lora_7b, text)

                answer_length_output_bloomz_7b1 = calculate_answer_length(answer_output_bloomz_7b1)
                proximity_output_bloomz_7b1 = calculate_proximity(question_squad, answer_output_bloomz_7b1, text)
                rouge_score_output_bloomz_7b1 = calculate_rouge_score(answer_output_bloomz_7b1, text)

                subset_data_point = {
                    "question_squad": question_squad,
                    "text": text,
                    "answer_output_llama-7b": answer_output_llama_7b,
                    "answer_output_alpaca-lora-7b": answer_output_alpaca_lora_7b,
                    "answer_output_bloomz-7b1": answer_output_bloomz_7b1,
                    "answer_length_output_llama_7b": answer_length_output_llama_7b,
                    "proximity_output_llama_7b": proximity_output_llama_7b,
                    "rouge_score_output_llama_7b": rouge_score_output_llama_7b,
                    "answer_length_output_alpaca_lora_7b": answer_length_output_alpaca_lora_7b,
                    "proximity_output_alpaca_lora_7b": proximity_output_alpaca_lora_7b,
                    "rouge_score_output_alpaca_lora_7b": rouge_score_output_alpaca_lora_7b,
                    "answer_length_output_bloomz_7b1": answer_length_output_bloomz_7b1,
                    "proximity_output_bloomz_7b1": proximity_output_bloomz_7b1,
                    "rouge_score_output_bloomz_7b1": rouge_score_output_bloomz_7b1,
                }

                processed_dataset.append(subset_data_point)

    return processed_dataset

# Read the dataset file and create the subset
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
            # Handle invalid JSON data here if needed
            print("Invalid JSON data:", file_path)

    # Process the dataset
    return create_subset(dataset)

def process_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder
    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_name = file_name.replace("cleaned", "evaluated")
        output_file_path = os.path.join(output_folder, output_file_name)

        # Read the dataset file and create the subset
        subset = read_dataset_file(input_file_path)

        # Write the sorted dataset to the subset file
        with open(output_file_path, 'w') as f:
            json.dump(subset, f, indent=2)

# Set the paths to the input and output folders
input_folder = "group_5_cleaned"
output_folder = "group_5_evaluated"

# Process the files in the input folder and save the output in the output folder
process_files(input_folder, output_folder)
