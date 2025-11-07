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
- config_name: 1_plotQA
  data_files: json/1_plotQA.json
- config_name: 2_needle
  data_files: json/2_needle.json
- config_name: 3_ego
  data_files: json/3_ego.json
- config_name: 4_count
  data_files: json/4_count.json
- config_name: 5_order
  data_files: json/5_order.json
- config_name: 6_anomaly_reco
  data_files: json/6_anomaly_reco.json
- config_name: 7_topic_reasoning
  data_files: json/7_topic_reasoning.json
language:
- en
size_categories:
- 1K<n<10K
---