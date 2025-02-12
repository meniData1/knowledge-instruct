# Repository Overview

This repository contains datasets and scripts related to the paper: 

**Knowledge-Instruct:
Effective Continual Pre-training from Limited Data using Instructions**.

## Folder Structure

- **`data/`** - Contains the datasets:
  - `fictional_companies_questions_open_end.jsonl`
  - `popqa_v2.jsonl`
    
  The `context` columns within these files contain the actual underlying data.

- **`create_fictional_companies.py`** - A script that outlines the process of generating the `companies` dataset underlying data.
- **`generate_open_end_questions.py`** - A script that outlines the process of generating the `companies` dataset questions and answers.

## Key Details

- **Companies Dataset**: Contains fictional company data, including question-answer pairs for reference.
- **PopQA**: This dataset is derived from the original PopQA but retains only tail knowledge.


##  Usage

To generate the `companies` dataset, first modify `LLMOpenAI` to your own version, then run:

```
python create_fictional_companies.py
```

Similarly, after changing `LLMOpenAI`, create question answer pairs by running:
```
python generate_open_end_questions.py
```
