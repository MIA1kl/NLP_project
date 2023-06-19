import json

def clean_json(json_data):
    cleaned_data = {}
    questions = {}
    answers = {}

    # Extract questions and answers
    for key, value in json_data.items():
        if key.startswith("QUESTION-"):
            question_key = key.replace("QUESTION-", "")
            questions[question_key] = value.strip()
        elif key.startswith("ANSWER-"):
            answer_key = key.replace("ANSWER-", "")
            answers[answer_key] = value.strip()

    # Combine questions and answers into cleaned data
    for key in questions.keys():
        cleaned_data[key] = {
            f"QUESTION-{key}": questions[key],
            f"ANSWER-{key}": answers.get(key, "")
        }

    return cleaned_data

# Load the JSON data
with open("cleaned_dataset/qa_pairs-cleaned-final.json", "r") as file:
    json_data = json.load(file)

# Clean the JSON data
cleaned_data = clean_json(json_data)

# Write cleaned data to a new JSON file
with open("output.json", "w") as file:
    json.dump(cleaned_data, file, indent=4)