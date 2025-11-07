import os
import glob
import json

'''
Step 1: 找到 ./json/ 目录下所有文件
Step 2: 遍历每个文件，读取内容
内容的形式大概是：
[
    {
        "video": "00000000",
        "type": "cartoon",
        "question": "What year appears in the opening caption of the video?",
        "candidates": [
            "(A) 1636",
            "(B) 1366",
            "(C) 1363",
            "(D) 1633"
        ],
        "answer": "(D) 1633",
        "question_type": [
            "key information retrieval"
        ],
        "time_reference": "00:15-00:19"
    },
    {
        "video": "00000000",
        "type": "cartoon",
        "question": "How is the weather like in the opening?",
        "candidates": [
            "(A) Cloudy",
            "(B) Snowy",
            "(C) Sunny",
            "(D) Rainy"
        ],
        "answer": "(B) Snowy",
        "question_type": [
            "event understanding"
        ],
        "time_reference": "00:00-01:16"
    },
......

Step 3:
检查 video 是否存在
对于 "video": "xxxxxxxx", 检查 ../../benchmark_data/lvbench_highres/xxxxxxxx.mp4 是否存在

Step 4:
把 step 3 中所有存在的数据保留，并把内容保存到 json_good/?????.json，文件名保持一致

Step 5:
最后打印统计数据，有多少个视频存在，有多少个视频不存在（注意多条数据可能对应同一个视频）
以及统计多少条数据对应的视频存在，多少条数据对应的视频不存在
'''

json_dir = "./json"
video_dir = "../../benchmark_data/lvbench_highres"
output_dir = "json_good"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_json_files = glob.glob(os.path.join(json_dir, "*.json"))
videos_exist = set()
videos_missing = set()
count_exist = 0
count_missing = 0

for file_path in all_json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = []
    for item in data:
        mp4_path = os.path.join(video_dir, item["video"] + ".mp4")
        if os.path.exists(mp4_path):
            filtered.append(item)
            videos_exist.add(item["video"])
            count_exist += 1
        else:
            videos_missing.add(item["video"])
            count_missing += 1
    out_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(filtered, out_f, ensure_ascii=False, indent=2)

print("Videos found:", len(videos_exist))
print("Videos missing:", len(videos_missing))
print("Data lines with existing video:", count_exist)
print("Data lines with missing video:", count_missing)