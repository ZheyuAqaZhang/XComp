---
license: mit
dataset_info:
- config_name: mc_question
  features:
  - name: video_name
    dtype: string
  - name: question
    dtype: string
  - name: question_id
    dtype: string
  - name: options
    sequence: string
  splits:
  - name: test
    num_bytes: 2009185
    num_examples: 11528
  download_size: 161062
  dataset_size: 2009185
- config_name: mc_question_val
  features:
  - name: video_name
    dtype: string
  - name: question
    dtype: string
  - name: question_id
    dtype: string
  - name: options
    sequence: string
  - name: answer_id
    dtype: string
  - name: area
    dtype: string
  - name: reasoning
    dtype: string
  - name: tag
    sequence: string
  splits:
  - name: validation
    num_bytes: 4676415
    num_examples: 19140
  download_size: 313591
  dataset_size: 4676415
configs:
- config_name: mc_question
  data_files:
  - split: test
    path: mc_question/test-*
- config_name: mc_question_val
  data_files:
  - split: validation
    path: mc_question_val/validation-*
---
