# from datasets import load_dataset

# load_dataset("json/lvbench.json")

import pyarrow.json as paj

paj.read_json("/mnt/petrelfs/share_data/lixinhao/lmms-eval_data/MVBench_mini/json/action_antonym.json")