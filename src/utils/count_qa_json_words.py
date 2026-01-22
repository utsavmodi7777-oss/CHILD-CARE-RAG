import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "../../new_data/qa_pairs/qa_pairs.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

total_query_words = 0
total_answer_words = 0

for i, obj in enumerate(data, start=1):
    query_words = len(obj["query"].split())
    answer_words = len(obj["ground_truth_answer"].split())
    total_words = query_words + answer_words
    
    total_query_words += query_words
    total_answer_words += answer_words
    
    # print(f"Object {i}: Query = {query_words} words, "
    #       f"Ground Truth Answer = {answer_words} words, "
    #       f"Total = {total_words}")

print(f"Total Query Words: {total_query_words}")
print(f"Total Ground Truth Answer Words: {total_answer_words}")
print(f"Overall Total Words: {total_query_words + total_answer_words}")
