# [NeurIPS 2025] One Token per Highly Selective Frame: Towards Extreme Compression for Long Video Understanding

<!-- todo: link to pdf ./statics/_Neurips__Camera_Ready__1_Token_Per_Frame_Compression.pdf -->
<!-- todo: link to video presentation https://recorder-v3.slideslive.com/?share=106573&s=bb94a71a-1e83-478a-8f93-607e868c8bef -->

- 📄 PDF：[`neurips_camera_ready_1_token_per_frame_compression.pdf`](https://openreview.net/pdf?id=bythzT0b81)
- 🎥 SlidesLive：[Link to Presentation](https://recorder-v3.slideslive.com/?share=106573&s=bb94a71a-1e83-478a-8f93-607e868c8bef)

Enhancing long video understanding via extreme compression by progressively reducing each selected frame to a single token.

<img width="952" height="487" alt="image" src="https://github.com/user-attachments/assets/fa5e3adf-34f9-4ea4-a5ff-bd9a2b390a65" />


## TLDR

Progressively compress video tokens to one token per frame. Achieve more comprehence long video understanding.

## Experiment

XComp is a fine-tuned model from VideoChat-Flash-2B. The environment and the data are the same. Please refer to [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash) for installation and data preparation.

- Training `./llava-train_videochat`
- Evaluate `./lmms-eval_videochat`

Download model parameters: [Google Drive](https://drive.google.com/file/d/1fubZjI640E2JZKWooX0xC2CsV8aCQyRh/view?usp=sharing), save to `XComp/llava-train_videochat/checkpoints/baseline_1000frame_cos/stagesuf-umt-hd-large-tome16_mlp_hd64_Qwen2_5_1_5B_stage3_short-long_mix_sft_mid2.yaml/`

## Citation

```
@inproceedings{
zhang2025one,
title={One Token per Highly Selective Frame: Towards Extreme Compression for Long Video Understanding},
author={Zheyu Aqa Zhang and Ziqi Pang and Shixing Chen and Xiang Hao and Vimal Bhat and Yu-Xiong Wang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=bythzT0b81}
}
```

## Acknowledgement

This work was supported in part by Amazon, NSF under Grants 2106825 and 2519216, and the DARPA Young Faculty Award. This work used computational resources, including Amazon Web Services (AWS), and the NCSA Delta and DeltaAI supercomputers through allocation CIS230012 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services \& Support (ACCESS) program.

We gratefully acknowledge the open-source projects that form the foundation of XComp: [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash), [Qwen](https://github.com/QwenLM/Qwen), and [LLaVA-Video](https://github.com/LLaVA-VL/LLaVA-NeXT).

We also thank the open-source of relevant projects: UMT, lmms-eval, transformers, ToMe, PyramidDrop, LongVideoBench, MLVU, VideoMME, and LVBench.