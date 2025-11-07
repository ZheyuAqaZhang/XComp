---
language:
- en
library_name: transformers
license: apache-2.0
metrics:
- accuracy
tags:
- multimodal
pipeline_tag: video-text-to-text
model-index:
- name: VideoChat-Flash-Qwen2_5-1_5B_res448
  results:
  - task:
      type: multimodal
    dataset:
      name: MLVU
      type: mlvu
    metrics:
    - type: accuracy
      value: 65.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MVBench
      type: mvbench
    metrics:
    - type: accuracy
      value: 70.0
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: Perception Test
      type: percepTest
    metrics:
    - type: accuracy
      value: 70.5
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LongVideoBench
      type: longvideobench
    metrics:
    - type: accuracy
      value: 58.3
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: VideoMME (wo sub)
      type: videomme
    metrics:
    - type: accuracy
      value: 57.0
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LVBench
      type: lvbench
    metrics:
    - type: accuracy
      value: 42.9
      name: accuracy
      verified: true


---

# 🦜VideoChat-Flash-Qwen2_5-2B_res448⚡
[\[📰 Blog\]](https://internvideo.github.io/blog/2024-12-31-VideoChat-Flash) [\[📂 GitHub\]](https://github.com/OpenGVLab/VideoChat-Flash)   [\[📜 Tech Report\]](https://www.arxiv.org/abs/2501.00574) [\[🗨️ Chat Demo\]](https://huggingface.co/spaces/OpenGVLab/VideoChat-Flash)

VideoChat-Flash-2B is constructed upon UMT-L (300M) and Qwen2.5-1.5B, employing only **16 tokens per frame**. By leveraging Yarn to extend the context window to 128k (Qwen2's native context window is 32k), our model supports input sequences of up to approximately **10,000 frames**. 

> Note: Due to a predominantly English training corpus, the model only exhibits basic Chinese comprehension, to ensure optimal performance, using English for interaction is recommended.



## 📈 Performance
| Model |  MVBench | LongVideoBench |  VideoMME(w/o sub)| Max Input Frames|
| ---   |  ---     |   ---            | ---     |  ---     | 
|[VideoChat-Flash-Qwen2_5-2B@448](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448)| 70.0 | 58.3   | 57.0| 10000 |
|[VideoChat-Flash-Qwen2-7B@224](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res224) | 73.2 | 64.2 | 64.0 | 10000 |
|[VideoChat-Flash-Qwen2_5-7B-1M@224](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B-1M_res224) | 73.4 | **66.5** | 63.5 | 50000 | 
|[VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B@224](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B) | **74.3** | 64.5 | 65.1 | 10000 |
|[VideoChat-Flash-Qwen2-7B@448](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448)| 74.0| 64.7 | **65.3**| 10000 |


## 🚀 How to use the model

First, you need to install [flash attention2](https://github.com/Dao-AILab/flash-attention) and some other modules. We provide a simple installation example below:
```
pip install transformers==4.40.1
pip install timm
pip install av
pip install imageio
pip install decord
pip install opencv-python
# optional
pip install flash-attn --no-build-isolation
```
Then you could use our model:
```python
from transformers import AutoModel, AutoTokenizer
import torch

# model setting
model_path = 'OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
image_processor = model.get_vision_tower().image_processor

mm_llm_compress = False # use the global compress or not
if mm_llm_compress:
    model.config.mm_llm_compress = True
    model.config.llm_compress_type = "uniform0_attention"
    model.config.llm_compress_layer_list = [4, 18]
    model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
else:
    model.config.mm_llm_compress = False

# evaluation setting
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)

video_path = "your_video.mp4"

# single-turn conversation
question1 = "Describe this video in detail."
output1, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=question1, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)

print(output1)

# multi-turn conversation
question2 = "How many people appear in the video?"
output2, chat_history = model.chat(video_path=video_path, tokenizer=tokenizer, user_prompt=question2, chat_history=chat_history, return_history=True, max_num_frames=max_num_frames, generation_config=generation_config)

print(output2)
```

## ✏️ Citation

```bibtex

@article{li2024videochatflash,
  title={VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling},
  author={Li, Xinhao and Wang, Yi and Yu, Jiashuo and Zeng, Xiangyu and Zhu, Yuhan and Huang, Haian and Gao, Jianfei and Li, Kunchang and He, Yinan and Wang, Chenting and others},
  journal={arXiv preprint arXiv:2501.00574},
  year={2024}
}

```