---
license: cc-by-nc-sa-4.0
extra_gated_prompt: >-
  The LongVideoBench dataset contains links to web videos for data collection
  purposes. LongVideoBench does not own the content linked within this dataset;
  all rights and copyright belong to the respective channel owners. Ensuring
  compliance with platform terms and conditions is the responsibility of these
  source channels. By accessing this dataset, you acknowledge and agree to the
  following terms:
extra_gated_fields:
  I understand that LongVideoBench does not own the videos in this dataset: checkbox
  I understand that LongVideoBench is not the creator of the videos in this dataset: checkbox
  I understand that, LongVideoBench may modify/delete its contents subject to the requirements of the creators or source platforms: checkbox
  I agree to use this dataset for non-commercial use ONLY: checkbox
  I agree with the data license (CC-BY-NC-SA 4-0) for this dataset: checkbox
task_categories:
- multiple-choice
- visual-question-answering
language:
- en
tags:
- long video understanding
- long context
- multimodal
- neurips 2024
pretty_name: longvideobench
---


![](https://github.com/longvideobench/longvideobench.github.io/blob/main/logo.png?raw=true)


# Dataset Card for LongVideoBench

<!-- Provide a quick summary of the dataset. -->




Large multimodal models (LMMs) are handling increasingly longer and more complex inputs. However, few public benchmarks are available to assess these advancements. To address this, we introduce LongVideoBench, a question-answering benchmark with video-language interleaved inputs up to an hour long. It comprises 3,763 web-collected videos with subtitles across diverse themes, designed to evaluate LMMs on long-term multimodal understanding.

The main challenge that LongVideoBench targets is to accurately retrieve and reason over detailed information from lengthy inputs. We present a novel task called referring reasoning, where questions contain a referring query that references related video contexts, requiring the model to reason over these details.

LongVideoBench includes 6,678 human-annotated multiple-choice questions across 17 categories, making it one of the most comprehensive benchmarks for long-form video understanding. Evaluations show significant challenges even for advanced proprietary models (e.g., GPT-4o, Gemini-1.5-Pro, GPT-4-Turbo), with open-source models performing worse. Performance improves only when models process more frames, establishing LongVideoBench as a valuable benchmark for future long-context LMMs.


## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

- **Curated by:** LongVideoBench Team
- **Language(s) (NLP):** English
- **License:** CC-BY-NC-SA 4.0

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** [https://github.com/longvideobench/LongVideoBench](https://github.com/longvideobench/LongVideoBench)
- **Homepage:** [https://longvideobench.github.io](https://longvideobench.github.io)
- **Leaderboard:** [https://huggingface.co/spaces/longvideobench/LongVideoBench](https://huggingface.co/spaces/longvideobench/LongVideoBench)

## Leaderboard (until Oct. 14, 2024)

We rank models by Test Total Performance. 

| Model | Test Total (5341) | Test 8s-15s | Test 15s-60s | Test 180s-600s | Test 900s-3600s | Val Total (1337) |
| --- | --- | --- | --- | --- | --- | --- |
| [GPT-4o (0513) (256)](https://platform.openai.com/docs/models/gpt-4o) | 66.7 | 71.6 | 76.8 | 66.7 | 61.6 | 66.7 |
| [Aria (256)](https://huggingface.co/rhymes-ai/Aria) | 65.0 | 69.4 | 76.6 | 64.6 | 60.1 | 64.2 |
| [LLaVA-Video-72B-Qwen2 (128)](https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2) | 64.9 | 72.4 | 77.4 | 63.9 | 59.3 | 63.9 |
| [Gemini-1.5-Pro (0514) (256)](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-1.5-pro-001) | 64.4 | 70.2 | 75.3 | 65.0 | 59.1 | 64.0 |
| [LLaVA-OneVision-QWen2-72B-OV (32)](https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov) | 63.2 | 74.3 | 77.4 | 61.6 | 56.5 | 61.3 |
| [LLaVA-Video-7B-Qwen2 (128)](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) | 62.7 | 69.7 | 76.5 | 62.1 | 56.6 | 61.1 |
| [Gemini-1.5-Flash (0514) (256)](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-1.5-flash-001) | 62.4 | 66.1 | 73.1 | 63.1 | 57.3 | 61.6 |
| [GPT-4-Turbo (0409) (256)](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4) | 60.7 | 66.4 | 71.1 | 61.7 | 54.5 | 59.1 |
| [InternVL2-40B (16)](https://huggingface.co/OpenGVLab/InternVL2-40B) | 60.6 | 71.4 | 76.6 | 57.5 | 54.4 | 59.3 |
| [GPT-4o-mini (250)](https://platform.openai.com/docs/models/gpt-4o-mini) | 58.8 | 66.6 | 73.4 | 56.9 | 53.4 | 56.5 |
| [MiniCPM-V-2.6 (64)](https://huggingface.co/openbmb/MiniCPM-V-2_6) | 57.7 | 62.5 | 69.1 | 54.9 | 49.8 | 54.9 |
| [Qwen2-VL-7B (256)](https://huggingface.co/openbmb/MiniCPM-V-2_6) | 56.8 | 60.1 | 67.6 | 56.7 | 52.5 | 55.6 |
| [Kangaroo (64)](https://huggingface.co/KangarooGroup/kangaroo) | 54.8 | 65.6 | 65.7 | 52.7 | 49.1 | 54.2 |
| [PLLaVA-34B (32)](https://github.com/magic-research/PLLaVA) | 53.5 | 60.1 | 66.8 | 50.8 | 49.1 | 53.2 |
| [InternVL-Chat-V1-5-26B (16)](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) | 51.7 | 61.3 | 62.7 | 49.5 | 46.6 | 51.2 |
| [LLaVA-Next-Video-34B (32)](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) | 50.5 | 57.6 | 61.6 | 48.7 | 45.9 | 50.5 |
| [Phi-3-Vision-Instruct (16)](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | 49.9 | 58.3 | 59.6 | 48.4 | 45.1 | 49.6 |
| [Idefics2 (16)](https://huggingface.co/HuggingFaceM4/idefics2-8b) | 49.4 | 57.4 | 60.4 | 47.3 | 44.7 | 49.7 |
| [Mantis-Idefics2 (16)](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2) | 47.6 | 56.1 | 61.4 | 44.6 | 42.5 | 47.0 |
| [LLaVA-Next-Mistral-7B (8)](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | 47.1 | 53.4 | 57.2 | 46.9 | 42.1 | 49.1 |
| [PLLaVA-13B (32)](https://github.com/magic-research/PLLaVA) | 45.1 | 52.9 | 54.3 | 42.9 | 41.2 | 45.6 |
| [InstructBLIP-T5-XXL (8)](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | 43.8 | 48.1 | 50.1 | 44.5 | 40.0 | 43.3 |
| [Mantis-BakLLaVA (16)](https://huggingface.co/TIGER-Lab/Mantis-bakllava-7b) | 43.7 | 51.3 | 52.7 | 41.1 | 40.1 | 43.7 |
| [BLIP-2-T5-XXL (8)](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | 43.5 | 46.7 | 47.4 | 44.2 | 40.9 | 42.7 |
| [LLaVA-Next-Video-M7B (32)](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) | 43.5 | 50.9 | 53.1 | 42.6 | 38.9 | 43.5 |
| [LLaVA-1.5-13B (8)](https://huggingface.co/llava-hf/llava-1.5-13b-hf) | 43.1 | 49.0 | 51.1 | 41.8 | 39.6 | 43.4 |
| [ShareGPT4Video (16)](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4Video) | 41.8 | 46.9 | 50.1 | 40.0 | 38.7 | 39.7 |
| [VideoChat2 (Mistral-7B) (16)](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) | 41.2 | 49.3 | 49.3 | 39.0 | 37.5 | 39.3 |
| [LLaVA-1.5-7B (8)](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | 40.4 | 45.0 | 47.4 | 40.1 | 37.0 | 40.3 |
| [mPLUG-Owl2 (8)](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) | 39.4 | 49.4 | 47.3 | 38.7 | 34.3 | 39.1 |
| [PLLaVA-7B (32)](https://github.com/magic-research/PLLaVA) | 39.2 | 45.3 | 47.3 | 38.5 | 35.2 | 40.2 |
| [VideoLLaVA (8)](https://github.com/PKU-YuanGroup/Video-LLaVA/) | 37.6 | 43.1 | 44.6 | 36.4 | 34.4 | 39.1 |
| [VideoChat2 (Vicuna 7B) (16)](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) | 35.1 | 38.1 | 40.5 | 33.5 | 33.6 | 36.0 |


## Uses

<!-- Address questions around how the dataset is intended to be used. -->

1. Download the dataset via Hugging Face Client:

```shell
huggingface-cli download longvideobench/LongVideoBench --repo-type dataset --local-dir LongVideoBench --local-dir-use-symlinks False
```

2. Extract from the `.tar` files:

```shell
cat videos.tar.part.* > videos.tar
tar -xvf videos.tar
tar -xvf subtitles.tar
```

3. Use the [LongVideoBench] dataloader to load the data from raw MP4 files and subtitles:

- (a) Install the dataloader:

```shell
git clone https://github.com/LongVideoBench/LongVideoBench.git
cd LongVideoBench
pip install -e .
```
- (b) Load the dataset in python scripts:

```python
from longvideobench import LongVideoBenchDataset

# validation
dataset = LongVideoBenchDataset(YOUR_DATA_PATH, "lvb_val.json", max_num_frames=64)

# test
dataset = LongVideoBenchDataset(YOUR_DATA_PATH, "lvb_test_wo_gt.json", max_num_frames=64)

print(dataset[0]["inputs"]) # A list consisting of PIL.Image and strings.
```

The "inputs" are interleaved video frames and text subtitles, followed by questions and option prompts. You can then convert them to the format that your LMMs can accept.


### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset is meant to evaluate LMMs on video understanding and long-context understanding abilities.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

We do not advise to use this dataset for training.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

- `lvb_val.json`: Validation set annotations.

- `lvb_test_wo_gt.json`: Test set annotations. Correct choice is not provided.

- `videos.tar.*`: Links to Videos.

- `subtitles.tar`: Links to Subtitles.


## Dataset Card Contact

haoning001@e.ntu.edu.sg


```
@misc{wu2024longvideobenchbenchmarklongcontextinterleaved,
      title={LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding}, 
      author={Haoning Wu and Dongxu Li and Bei Chen and Junnan Li},
      year={2024},
      eprint={2407.15754},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.15754}, 
}
```