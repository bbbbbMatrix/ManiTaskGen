

01_preprocessings.py 

input: 
data/datasets/replica_dataset 
data/templates/renaming_dict.json

output: 
data/cache/dataset_parsed.json
data/cache/scene_graph.pkl 
data/cache/rename_dict.json 
data/images/image4rename/...

02a_processed_based_tasks.py
input:
data/cache/scene_graph.pkl
data/cache/rename_dict.json

output:
data/cache/process_based_tasks.pkl 
data/cache/process_based_tasks_ins.txt

02b_outcome_based_tasks.py
input:
data/cache/scene_graph.pkl
data/cache/rename_dict.json 
data/templates/voting_prompts.json
output:
data/cache/outcome_based_tasks_ins.txt
data/images/image4outcomebase/

03a_benchmark_executor.py
input:
data/cache/scene_graph.pkl 
data/cache/rename_dict.json
data/cache/process_based_tasks.pkl
data/templates/benchmark_prompts.json
data/templates/reflection_prompts.json
output:
data/images/image4interaction/..
data/cache/reflection_notes.txt
data/results/benchmark_results.txt


03b_benchmark_executor.py
input:
data/cache/scene_graph.pkl 
data/cache/rename_dict.json
data/cache/reflection_notes.txt
data/templates/benchmark_prompts.json
data/templates/reflection_prompts.json
output:
data/images/image4interaction/..
data/cache/reflection_notes.txt
data/results/benchmark_results.txt


此外，还需要编写一些bash脚本来方便用户运行这些python脚本。可以连续运行的包括01+02a+03a, 01+02a+03b, 01+02b. 