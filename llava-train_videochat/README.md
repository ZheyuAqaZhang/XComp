# How to train XComp?

### Step 1. Install

Follow <a href='https://github.com/OpenGVLab/VideoChat-Flash/blob/2ce999bdcdcfb771cc93986c19f346bd20f0970a/llava-train_videochat/README.md'>[VideoChat-Flash Training Guideline]</a> set up training evironment.

### Step 2. Modify transformers

Clone transformers 4.40 and replace `transformer_qwen2` by `transformer_qwen2_copy`. Then install the modified transformers.

### Step 3. Run script

The training script is `./scripts/XComp/train.sh`

Multi-GPUs with $\ge$ 80GB memory per GPU are preferred.