import spacy
from rouge import Rouge
import json


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Calculate answer length
def calculate_answer_length(answer):
    return len(answer.split())

# Calculate proximity between question words and answer in context
def calculate_proximity(question, answer, context):
    doc = nlp(context)
    question_tokens = nlp(question)
    answer_tokens = nlp(answer)
    
    # Calculate the proximity score
    proximity_score = any(question_token.text in answer_tokens for question_token in question_tokens)

    # Return the proximity score
    return proximity_score

# Calculate ROUGE score
def calculate_rouge_score(question, answer, reference):
    rouge = Rouge()
    scores = rouge.get_scores(answer, reference)
    
    # Return the ROUGE score
    return scores[0]["rouge-1"]["f"]

# Apply selection rules and create a subset
def create_subset(data_points, subset_path):
    processed_dataset = []
    for data_point in data_points:
        question = data_point["question_race"]
        answer = data_point["answer_race"]
        question_squad = data_point["question_squad"]
        answer_squad = data_point["answer_squad"]
        text = data_point["text"]
        
        answer_length = calculate_answer_length(answer)
        proximity = calculate_proximity(question, answer, text)
        rouge_score = calculate_rouge_score(question, answer, text)
        
        subset_data_point = {
            "question_race": question,
            "answer_race": answer,
            "question_squad": question_squad,
            "answer_squad": answer_squad,
            "text": text,
            "answer_length": answer_length,
            "proximity": proximity,
            "rouge_score": rouge_score
        }
        
        processed_dataset.append(subset_data_point)
    
    # Sort the dataset based on the selection rules
    sorted_dataset = sorted(
        processed_dataset,
        key=lambda x: (x["answer_length"], -x["proximity"], -x["rouge_score"]),
        reverse=True
    )
    
    # Write the sorted dataset to the subset file
    with open(subset_path, 'w') as f:
        json.dump(sorted_dataset, f, indent=4)

# Read the dataset file and create the subset
def read_dataset_file(file_path, chunk_size=1000):
    dataset = []
    with open(file_path, "r") as file:
        chunk = []
        for line in file:
            try:
                data_point = json.loads(line)
                chunk.append(data_point)
                
                # Process the chunk if it reaches the specified size
                if len(chunk) >= chunk_size:
                    create_subset(chunk, subset_file_path)
                    chunk = []
            except json.JSONDecodeError:
                # Handle invalid JSON data here if needed
                print("Invalid JSON data:", line)
        
        # Process the remaining chunk
        if chunk:
            create_subset(chunk, subset_file_path)
    
    return dataset

# Specify the path to your dataset file
dataset_file_path = "cleaned_dataset/qa_pairs-cleaned-final.jsonl"

# Specify the path for the subset file
subset_file_path = "subset_dataset.json"

# Read the dataset file and create the subset
subset = read_dataset_file(dataset_file_path)

# Print the subset
for data in subset:
    print(data)
