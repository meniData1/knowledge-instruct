#!/usr/bin/env python3
"""
Module: generate_questions.py

Description:
    This module loads a dataset of fictional companies, prepares context chunks,
    queries an LLM (via LLMOpenAI) to generate open-ended questions, processes the
    responses, and saves the results as a JSONL file.
"""

import json
import pandas as pd
import random
import pickle
import ast
import sys
import os
from tqdm import tqdm


# Append parent directory to system path if needed
sys.path.append(os.path.join(os.getcwd(), '..'))

# Replace with your own LLM class
from utils.open_ai_utils import LLMOpenAI

# Set random seed for reproducibility
random.seed(42)
tqdm.pandas()

# Constants
CHUNK_SIZE = 10
NUM_QUESTIONS_PER_CHUNK = 10

# Prompts
META_PROMPT = """
Context information is below.\n
Given the context information and no prior knowledge.\n
Generate only open-ended questions based on the below query.\n
The context of each question should be easily inferred by reading the question. If you are referring to a specific event, mention the exact name and date of the event.\n
Make sure that:
 1. Each question can be answered correctly by someone who has past exposure to the context but does not currently have the context available.
 2. The questions are not ambiguous in any way, having a single correct and specific answer.
 3. The correct answer cannot be deduced solely based on reasoning. Only someone who has been exposed to the context should be able to answer the question.
 4. The correct answer is not ambiguous in any way. Clarify the expected format of the answer if necessary.
 5. Each question is an atomic question. Do not ask multiple questions in a single question.

You are a Teacher/Professor. Your task is to set up 
{num_questions_per_chunk} open-ended questions for an upcoming
quiz/examination. The questions should be diverse in nature
across the document. Do not repeat the same question twice. Restrict the questions to the
context information provided.
Return a JSON formatted string with the following fields:
question, correct_answer. 
"""

CONTEXT_PROMPT = """---------------------\n{context_str}\n---------------------\n
Return a valid python list of the question jsons.
"""

SAMPLING_PARAMS = {
    "temperature": 0.5,
    "max_tokens": 2000
}


def load_company_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Melted DataFrame containing 'company' and 'text' columns.
    """
    df = pd.read_csv(csv_path)
    df.fillna("None", inplace=True)
    # Flatten all columns except 'company' so that there are only two columns: 'company' and 'text'
    df = df.melt(id_vars=['company'], var_name='column', value_name='text')
    return df


def prepare_corpus(df: pd.DataFrame) -> list:
    """
    Prepare a corpus of context chunks from the dataset.

    Args:
        df (pd.DataFrame): DataFrame with 'company' and 'text' columns.

    Returns:
        list: List of context strings.
    """
    corpus = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing corpus"):
        company = row['company']
        text = row['text']
        # Create a chunk with company and text context
        chunk = f"### Company: {company}.\n### Text: {text}"
        corpus.append(chunk)
    return corpus


def generate_corpus_messages(corpus: list) -> list:
    """
    Generate LLM messages for each context chunk.

    Args:
        corpus (list): List of context strings.

    Returns:
        list: List of message lists for each context chunk.
    """
    messages = []
    for chunk in tqdm(corpus, desc="Generating corpus messages"):
        message = [
            {
                "role": "system",
                "content": META_PROMPT.format(num_questions_per_chunk=NUM_QUESTIONS_PER_CHUNK)
            },
            {
                "role": "user",
                "content": CONTEXT_PROMPT.format(context_str=chunk)
            }
        ]
        messages.append(message)
    return messages


def generate_responses(llm: LLMOpenAI, corpus_messages: list, sampling_params: dict) -> list:
    """
    Generate responses from the LLM for the given messages.

    Args:
        llm (LLMOpenAI): The LLM instance.
        corpus_messages (list): List of message lists.
        sampling_params (dict): Sampling parameters for the LLM.

    Returns:
        list: List of responses from the LLM.
    """
    responses = llm.generate(corpus_messages, sampling_params, max_workers=30)
    return responses


def process_responses(corpus: list, responses: list) -> list:
    """
    Process the LLM responses to extract and update the question data.

    Args:
        corpus (list): List of context strings.
        responses (list): List of responses from the LLM.

    Returns:
        list: List of question dictionaries.
    """
    all_questions = []
    failures = 0

    for idx, (chunk, response) in enumerate(zip(corpus, responses)):
        # Extract JSON string portion from the response
        json_str = response[response.find("["): response.rfind("]") + 1]
        try:
            python_list = json.loads(json_str)
        except Exception:
            try:
                python_list = ast.literal_eval(json_str)
                # If python_list is not a dict and the first element is a dict, adjust accordingly
                if isinstance(python_list, list) and python_list and isinstance(python_list[0], dict):
                    python_list = python_list[0]
            except Exception:
                print(f"Failed to parse response at index {idx}")
                failures += 1
                continue

        try:
            for item in python_list:
                item['context'] = chunk
        except Exception:
            print(f"Failed to update context for response at index {idx}")
            failures += 1
            continue

        all_questions.extend(python_list)

    print(f"Failures: {failures}")
    print(f"Total questions: {len(all_questions)}")
    return all_questions


def add_company_and_format_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the company name from the context and format the question text.

    Args:
        df (pd.DataFrame): DataFrame containing the question data.

    Returns:
        pd.DataFrame: Updated DataFrame with the formatted question text.
    """
    # Extract company name from context
    df['company'] = df['context'].apply(lambda x: x.split("### Company: ")[1].split(':')[0])
    # Format the question text to include the company reference
    df['question'] = df.apply(
        lambda x: f'The question refers to the company "{x["company"]}".\n### Question: {x["question"]}',
        axis=1
    )
    df.drop(columns=['company'], inplace=True)
    return df


def save_questions_to_jsonl(df: pd.DataFrame, jsonl_path: str) -> None:
    """
    Save the questions DataFrame to a JSONL file.

    Args:
        df (pd.DataFrame): DataFrame containing question data.
        jsonl_path (str): Path to save the JSONL file.
    """
    df.to_json(jsonl_path, orient='records', lines=True, index=False)


def main():
    # Initialize the LLM
    llm = LLMOpenAI(model='gpt-4o')

    # Load the dataset
    csv_path = "fictional_companies_dataset.csv"
    df = load_company_dataset(csv_path)

    # Prepare the corpus of context chunks
    corpus = prepare_corpus(df)

    # Generate messages for each context chunk
    corpus_messages = generate_corpus_messages(corpus)

    # Generate responses from the LLM
    responses = generate_responses(llm, corpus_messages, SAMPLING_PARAMS)

    # Process the responses to extract questions
    all_questions = process_responses(corpus, responses)

    # Convert the questions list to a DataFrame
    questions_df = pd.DataFrame(all_questions)

    # Add company information and format the question text
    questions_df = add_company_and_format_question(questions_df)

    # Ensure the output directory exists and save the questions to a JSONL file
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    jsonl_path = os.path.join(data_dir, 'fictional_companies_questions_open_end_test.jsonl')
    save_questions_to_jsonl(questions_df, jsonl_path)

    # Print a sample of the output
    print(questions_df[['question', 'correct_answer']].sample(10).values)


if __name__ == "__main__":
    main()
