---
license: mit
extra_gated_prompt: >-
  You agree to not use the dataset to conduct experiments that cause harm to
  human subjects. Please note that the data in this dataset may be subject to
  other agreements. Before using the data, be sure to read the relevant
  agreements carefully to ensure compliant use. Video copyrights belong to the
  original video creators or platforms and are for academic research use only.
task_categories:
- visual-question-answering
extra_gated_fields:
  Name: text
  Company/Organization: text
  Country: text
  E-Mail: text
modalities:
- Video
- Text
configs:
- config_name: lvbench
  data_files: json/lvbench_clean.json
- config_name: lvbench_cartoon
  data_files: json/lvbench_clean_cartoon.json
- config_name: lvbench_documentary
  data_files: json/lvbench_clean_documentary.json
- config_name: lvbench_live
  data_files: json/lvbench_clean_live.json
- config_name: lvbench_selfmedia
  data_files: json/lvbench_clean_selfmedia.json
- config_name: lvbench_sport
  data_files: json/lvbench_clean_sport.json
- config_name: lvbench_tv
  data_files: json/lvbench_clean_tv.json
language:
- en
size_categories:
- 1K<n<10K
---