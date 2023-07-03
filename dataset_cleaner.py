import os
import json

input_folder = "group_5"  # Path to the folder containing the input files
output_folder = "group_5_cleaned"  # Path to the folder where you want to save the output files

files = [
    "qa_pairs-cleaned-biology-news.jsonl",
    "qa_pairs-cleaned-chemistry-news.jsonl",
    "qa_pairs-cleaned-earth-news.jsonl",
    "qa_pairs-cleaned-nanotech-news.jsonl",
    "qa_pairs-cleaned-physics-news.jsonl",
    "qa_pairs-cleaned-science-news.jsonl",
    "qa_pairs-cleaned-space-news.jsonl"
]

model_outputs = ["output_llama-7b", "output_alpaca-lora-7b", "output_bloomz-7b1"]

for file in files:
    input_file_path = os.path.join(input_folder, file)
    output_file_path = os.path.join(output_folder, file.replace(".jsonl", ".json"))
    cleaned_data_list = []
    
    with open(input_file_path, 'r') as f:
        for line in f:
            data_dict = json.loads(line)
            
            question_squad = data_dict["question_squad"]
            
            cleaned_data = {
                "question_squad": data_dict["question_squad"],
                "answer_squad": data_dict["answer_squad"],
                "text": data_dict["text"],
            }
            
            for model_output in model_outputs:
                if model_output in data_dict:
                    qa_pairs = data_dict[model_output].split("QUESTION:")
                    formatted_output = ""
                    for qa_pair in qa_pairs[1:]:
                        qa_pair_parts = qa_pair.strip().split("ANSWER:")
                        if len(qa_pair_parts) == 2:
                            question, answer = qa_pair_parts
                            if question.strip() == question_squad.strip():
                                formatted_output += answer.strip() + " "
                    cleaned_data[model_output] = formatted_output.strip()
            
            cleaned_data_list.append(cleaned_data)
    
    with open(output_file_path, 'w') as f:
        json.dump(cleaned_data_list, f, indent=2)
