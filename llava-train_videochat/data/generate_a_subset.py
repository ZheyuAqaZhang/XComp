'''
Step 1. load 以下三个dataset的配置文件，分别是
    1.1. data/stage1_init_connector_iv1m.yaml
        形如：
datasets:
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/smit_caption_481k.json
    sampling_strategy: all
    data_root: OpenGVLab/smit/S-MiT/
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/blip_laion_cc_sbu_558k.json
    sampling_strategy: all
    data_root: OpenGVLab/LLaVA-Pretrain/
...

    1.2. data/stage2_short_pretrain_iv6m.yaml
        形如：
datasets:
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-118K.json
    sampling_strategy: all
    data_root: OpenGVLab/stage2/LLaVA-ReCap-118K/data/extracted_images/
  # - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/webvid-fuse_caption_2m.json
  #   sampling_strategy: all
  #   data_root: https://github.com/m-bain/webvid
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-CC3M.json
    sampling_strategy: first:5%
    data_root: OpenGVLab/stage2/LLaVA-ReCap-CC3M/data/extracted_images/

    1.3. data/stage3_short-long_mix_sft.yaml
        也是差不多的格式

Step 2. 对于每个配置文件 data/xxx.yaml，都要生成一个文件 data/subset/xxx_subset.yaml，并把相关文件复制到 OpenGVLab/Subset/... 下
    2.1. 对于生成的 xxx_subset.yaml，要求保留每个未被注释的 dataset，但是要把 sampling_strategy 改为 first:?%
        具体来说，对于 stage1，需要把 all 改为 first:50%，把 first:?% 改为 first:0.5*?% （可以有小数）
        对于 stage2，需要把 all 改为 first:5%，把 first:?% 改为 first:0.05*?% （可以有小数）
        对于 stage3，需要把 all 改为 first:2%，把 first:?% 改为 first:0.02*?% （可以有小数）
    2.2. 对于涉及到的文件，你需要去读取 dataset 的 json 文件(！注意：或者 jsonl 文件，需要用另一种读取方式），并根据 first:?% 的要求，选取 json 文件的前 ?% 的数据
        对于每一条数据，如果它有 image 元素，则把 {data_root}/{image} 复制到 OpenGVLab/Subset/... 下 (注意，data_root 总是以 OpenGVLab 开头，所以目标路径是在第一级目录下再加一个 "Subset" 目录)
        类似地，如果它有 video 元素，则把 {data_root}/{video} 复制到 OpenGVLab/Subset/... 下
        （注意目标路径可能不存在，需要makedir）
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import yaml
import shutil

"""
本脚本用于：
1. 读取三个 stage 的配置文件 (data/stage1_init_connector_iv1m.yaml,
   data/stage2_short_pretrain_iv6m.yaml, data/stage3_short-long_mix_sft.yaml)。
2. 为每个配置文件生成一个对应的 "subset" 配置文件，例如：
   data/subset/stage1_init_connector_iv1m_subset.yaml
3. 按照要求：
   - 修改 sampling_strategy:
       stage1:
         all -> first:50%
         first:?% -> first:0.5 * ?%
       stage2:
         all -> first:5%
         first:?% -> first:0.05 * ?%
       stage3:
         all -> first:2%
         first:?% -> first:0.02 * ?%
   - 读取对应的 JSON / JSONL 文件（如果是被注释的 datasets 则跳过），
     截取前若干百分比数据后，写入新的 JSON / JSONL 到 "OpenGVLab/Subset/..." 路径下
   - 如果数据中有 "image" 字段，则复制对应的图像文件到 "OpenGVLab/Subset/..." 下
     如果有 "video" 字段，则复制对应的视频文件到 "OpenGVLab/Subset/..." 下
   - 更新新的配置文件中的 data_root 和 json_path 都指向 "OpenGVLab/Subset/..." 下的路径
   - 注意在复制文件时，需要保持对应的层级目录结构，如果目录不存在，需要先创建目录
"""

# 根据 stage 不同，对应的 "all" -> first:?% 以及 first:?% -> first:xx% 的转换参数
STAGE_CONFIG = {
    1: {
        "all": 50.0,   # all -> first:50%
        "factor": 0.5  # first:?% -> first:0.5 * ?%
    },
    2: {
        "all": 5.0,    # all -> first:5%
        "factor": 0.05 # first:?% -> first:0.05 * ?%
    },
    3: {
        "all": 2.0,    # all -> first:2%
        "factor": 0.02 # first:?% -> first:0.02 * ?%
    }
}

def parse_sampling_strategy(strategy: str, stage: int) -> str:
    """
    根据 stage，将原来的 sampling_strategy 转换为新的 sampling_strategy。
    - 如果是 'all' -> 'first:X%'
    - 如果是 'first:?%' -> 'first:(factor * ?)%'
    注意：这里 factor * ? 以后可能是浮点数，需要保留合理的小数。
    """
    # 如果是 'all'
    if strategy.strip() == "all":
        return f"first:{STAGE_CONFIG[stage]['all']}%"
    
    # 如果形如 'first:xxx%'，抽取数字部分
    match = re.match(r"^first:(\d+(\.\d+)?)%$", strategy.strip())
    if match:
        old_value = float(match.group(1))
        new_value = old_value * STAGE_CONFIG[stage]["factor"]
        # 结果可能是浮点数，这里示例中保留两位小数
        return f"first:{new_value:.2f}%"
    
    # 如果格式都对不上，原样返回或根据需求处理
    return strategy

def ensure_dir(path: str):
    """若目录不存在则创建目录。"""
    os.makedirs(path, exist_ok=True)

def copy_file_with_structure(src_root: str, rel_path: str):
    """
    将 src_root/rel_path 文件复制到目标：
        dst_root = 在 src_root 的 "OpenGVLab" 后面插入 "/Subset"
        即将 "OpenGVLab/xxx" 变成 "OpenGVLab/Subset/xxx"
    例如:
       src_root = "OpenGVLab/smit/S-MiT"
       rel_path = "images/001.png"
       => 源文件位置:   OpenGVLab/smit/S-MiT/images/001.png
       => 目标根路径:   OpenGVLab/Subset/smit/S-MiT
       => 目标完整文件: OpenGVLab/Subset/smit/S-MiT/images/001.png
    """
    # 找到 src_root 中 "OpenGVLab" 的第一次出现位置
    idx = src_root.find("OpenGVLab")
    if idx < 0:
        # 若未找到，理论上不应出现，可根据需要决定是否 raise
        raise ValueError(f"data_root 不以 OpenGVLab 开头: {src_root}")
    
    # 构造新的 subset 路径
    dst_root = src_root[:idx] + "OpenGVLab/Subset/" + src_root[idx+len("OpenGVLab/"):]
    # 目标文件完整路径
    dst_full_dir = os.path.join(dst_root, os.path.dirname(rel_path))
    ensure_dir(dst_full_dir)
    
    src_file = os.path.join(src_root, rel_path)
    dst_file = os.path.join(dst_root, rel_path)
    
    # 如果源文件存在，则复制；若不存在，可能需根据需求处理 (此处直接忽略或 raise)
    if os.path.exists(src_file) and os.path.isfile(src_file):
        shutil.copy2(src_file, dst_file)
    elif os.path.exists(src_file) and os.path.isdir(src_file):
        # 如果是目录，则复制该文件夹，若存在则覆盖
        shutil.copytree(src_file, dst_file, dirs_exist_ok=True)
    else:
        print(f"警告：源文件不存在，跳过复制: {src_file}")

def copy_and_generate_subset_json(json_path: str, data_root: str, new_sampling_strategy: str, stage: int) -> str:
    """
    读取 json_path (可能是 .json 或 .jsonl)，根据 new_sampling_strategy (形如 'first:xx%') 截取前 xx% 的数据，
    并将 image 或 video 指向的文件复制到新的路径下，最终生成新的 JSON/JSONL 文件路径，并返回它。
    
    返回值：新的 subset json/jsonl 文件在 OpenGVLab/Subset/... 下的完整路径
    """
    # (1) 根据 json_path 找到新的 subset 路径
    idx = json_path.find("OpenGVLab")
    if idx < 0:
        raise ValueError(f"json_path 不以 OpenGVLab 开头: {json_path}")

    # 新的 JSON/JSONL 路径
    new_json_path = json_path[:idx] + "OpenGVLab/Subset/" + json_path[idx+len("OpenGVLab/"):]
    
    # 确保目标目录存在
    ensure_dir(os.path.dirname(new_json_path))
    
    # (2) 解析 new_sampling_strategy，得到截取百分比
    match = re.match(r"^first:(\d+(\.\d+)?)%$", new_sampling_strategy.strip())
    if match:
        fraction = float(match.group(1)) / 100.0
    else:
        fraction = 1.0  # 如果异常，就默认用全部
    
    # (3) 读取数据 (json or jsonl)
    data_entries = []
    if json_path.endswith(".json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data_entries = json.load(f)
    elif json_path.endswith(".jsonl"):
        data_entries = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data_entries.append(json.loads(line))
    else:
        raise ValueError(f"只支持 .json / .jsonl 文件，当前文件类型无法处理: {json_path}")

    # (4) 截取前 fraction 的数据
    total_count = len(data_entries)
    subset_count = int(round(total_count * fraction))
    subset_entries = data_entries[:subset_count]

    # (5) 复制 subset 中的文件 (如果有 "image" 或 "video")
    for entry in subset_entries:
        if "image" in entry:
            if isinstance(entry["image"], list):
                for img in entry["image"]:
                    copy_file_with_structure(data_root, img)
            else:
                copy_file_with_structure(data_root, entry["image"])
        if "video" in entry:
            if isinstance(entry["video"], list):
                for vid in entry["video"]:
                    copy_file_with_structure(data_root, vid)
            else:
                copy_file_with_structure(data_root, entry["video"])

    # # (6) 将 subset_entries 写入到 new_json_path
    # if new_json_path.endswith(".json"):
    #     with open(new_json_path, "w", encoding="utf-8") as f:
    #         json.dump(subset_entries, f, ensure_ascii=False, indent=2)
    # else:
    #     # jsonl
    #     with open(new_json_path, "w", encoding="utf-8") as f:
    #         for item in subset_entries:
    #             f.write(json.dumps(item, ensure_ascii=False) + "\n")
    #
    # return new_json_path

def transform_config(
    config_file: str,
    stage: int,
    output_file: str
):
    """
    读取 config_file (YAML)，生成子集配置写到 output_file (YAML)。
    主要逻辑：
      - 对未被注释的 datasets，修改其 sampling_strategy
      - 生成对应的 subset 的 json / jsonl 文件
      - 修改 data_root 为 subset 路径
      - 修改 json_path 为 subset 路径
    """
    with open(config_file, "r", encoding="utf-8") as f:
        # 直接用 yaml.safe_load 会忽略掉 # 注释的 datasets
        # 因此，这里我们先读入 YAML，然后只保留真正反序列化出来的 datasets。
        config_data = yaml.safe_load(f)

    if "datasets" not in config_data:
        print(f"警告: {config_file} 中未包含 'datasets' 字段，跳过处理。")
        return
    
    new_datasets = []
    for ds in config_data["datasets"]:
        # ds 是一个 dict，比如:
        # {
        #    "json_path": "OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/smit_caption_481k.json",
        #    "sampling_strategy": "all",
        #    "data_root": "OpenGVLab/smit/S-MiT/"
        # }
        # 不考虑注释了 (因为yaml.safe_load已经不会把注释load进来了)

        # 如果 file data/subset/done.json 存在，并且包含 ds["json_path"]，则跳过
        ready_to_copy_files = True
        done = {}
        if os.path.exists("data/subset/done.json"):
            with open("data/subset/done.json", "r", encoding="utf-8") as f:
                done = json.load(f)
        if ds["json_path"] in done:
            print(f"跳过已处理的数据集: {ds['json_path']}")
            ready_to_copy_files = False
        
        old_strategy = ds.get("sampling_strategy", "all")
        new_strategy = parse_sampling_strategy(old_strategy, stage)

        if "data_root" not in ds:
            ready_to_copy_files = False
        else:
            old_data_root = ds["data_root"]
            # 更新 data_root -> Insert "/Subset/"
            idx = old_data_root.find("OpenGVLab")
            if idx < 0:
                raise ValueError(f"data_root 不以 OpenGVLab 开头: {old_data_root}")
            new_data_root = old_data_root[:idx] + "OpenGVLab/Subset/" + old_data_root[idx+len("OpenGVLab/"):]

            old_json_path = ds["json_path"]
            # 先生成并写入新 json / jsonl
            if ready_to_copy_files:
                copy_and_generate_subset_json(old_json_path, old_data_root, new_strategy, stage)
                done[old_json_path] = True
                with open("data/subset/done.json", "w", encoding="utf-8") as f:
                    json.dump(done, f, indent=2)

        # 更新 new_json_path 也要指向 subset
        # 但它已经在函数中确定放在 /Subset/ 下了，所以只需要从 old_json_path 同样方式处理或者直接用返回值
        # 这里直接使用函数返回值
        ds_new = ds.copy()
        ds_new["sampling_strategy"] = new_strategy
        if "data_root" in ds_new:
            ds_new["data_root"] = new_data_root
        new_datasets.append(ds_new)

    # 组装新的 config
    new_config = {
        "datasets": new_datasets
    }

    # 写入新的 YAML
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, sort_keys=False, allow_unicode=True)

def main():
    # 需要处理的三个输入文件及对应的 stage
    stage_files = [
        ("data/stage1_init_connector_iv1m.yaml", 1),
        ("data/stage2_short_pretrain_iv6m.yaml", 2),
        ("data/stage3_short-long_mix_sft.yaml", 3),
    ]

    for (cfg_file, stage_id) in stage_files:
        base_name = os.path.basename(cfg_file)            # e.g. "stage1_init_connector_iv1m.yaml"
        name_only, _ = os.path.splitext(base_name)        # e.g. "stage1_init_connector_iv1m"
        out_file = os.path.join("data", "subset", f"{name_only}_subset.yaml")

        print(f"\n开始处理 {cfg_file} (stage={stage_id})...")
        transform_config(cfg_file, stage_id, out_file)
        print(f"完成: 生成子集配置 {out_file}")

if __name__ == "__main__":
    main()

