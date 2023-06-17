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
    answer_similarity = nlp(answer).similarity(doc)
    
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
def create_subset(data_points, subset_path):
    processed_dataset = []
    for data_point in data_points:
        question_race = data_point["question_race"]
        answer_race = data_point["answer_race"]
        question_squad = data_point["question_squad"]
        answer_squad = data_point["answer_squad"]
        text = data_point["text"]
        
        answer_length_race = calculate_answer_length(answer_race)
        proximity_race = calculate_proximity(question_race, answer_race, text)
        rouge_score_race = calculate_rouge_score(answer_race, text)
        
        answer_length_squad = calculate_answer_length(answer_squad)
        proximity_squad = calculate_proximity(question_squad, answer_squad, text)
        rouge_score_squad = calculate_rouge_score(answer_squad, text)
        
        subset_data_point = {
            "question_race": question_race,
            "answer_race": answer_race,
            "question_squad": question_squad,
            "answer_squad": answer_squad,
            "text": text,
            "answer_length_race": answer_length_race,
            "proximity_race": proximity_race,
            "rouge_score_race": rouge_score_race,
            "answer_length_squad": answer_length_squad,
            "proximity_squad": proximity_squad,
            "rouge_score_squad": rouge_score_squad,
        }
        
        processed_dataset.append(subset_data_point)
    
    # Sort the dataset based on the selection rules
    sorted_dataset = sorted(
        processed_dataset,
        key=lambda x: (x["answer_length_race"], -x["proximity_race"], -x["rouge_score_race"]),
        reverse=True
    )
    
    # Write the sorted dataset to the subset file
    with open(subset_path, 'w') as f:
        json.dump(sorted_dataset, f, indent=4)


# Read the dataset file and create the subset
def read_dataset_file(file_path, chunk_size=1000):
    dataset = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                data_point = json.loads(line)
                dataset.append(data_point)
                
                # Process the chunk if it reaches the specified size
                if len(dataset) >= chunk_size:
                    create_subset(dataset, subset_file_path)
                    dataset = []
            except json.JSONDecodeError:
                # Handle invalid JSON data here if needed
                print("Invalid JSON data:", line)
        
        # Process the remaining chunk
        if dataset:
            create_subset(dataset, subset_file_path)
    
    return dataset

# Specify the path to your dataset file
dataset_file_path = "cleaned_dataset/qa_pairs-cleaned-final.json"

# Specify the path for the subset file
subset_file_path = "subset_dataset.json"

# Read the dataset file and create the subset
subset = read_dataset_file(dataset_file_path)

# Read the subset file
with open(subset_file_path, 'r') as f:
    subset_data = json.load(f)

# Write the subset data to a new JSON file
output_file_path = "subset_output.json"
with open(output_file_path, 'w') as f:
    json.dump(subset_data, f, indent=4)
