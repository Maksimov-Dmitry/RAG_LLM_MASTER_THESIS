from ragas.llms.openai import OpenAI
import os
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    context_relevancy,
    faithfulness,
    answer_correctness,
    AnswerSimilarity,
)
from tqdm import tqdm
from time import sleep


def calculate_DocHitRate(dataset):
    hits = 0
    for row in dataset:
        if row['document'] in row['contexts_documents']:
            hits += 1
    return hits / len(dataset)


def calculate_HitRate(dataset):
    hits = 0
    for row in dataset:
        for i, doc in enumerate(row['contexts_documents']):
            if doc == row['document'] and row['page'] == row['contexts_pages'][i]:
                hits += 1
                break  # Break if a hit is found for this row
    return hits / len(dataset)


def retriever_llm_evaluate(dataset, model_name, max_trials=5, batch_size=10):
    model = OpenAI(model_name)
    context_precision.llm = model
    context_recall.llm = model
    context_relevancy.llm = model
    scores = {
        'context_precision': [],
        'context_recall': [],
        'context_relevancy': []
    }
    temp_dataset = dataset.map(lambda example: {'ground_truths': [example['ground_truth']]})
    for i in tqdm(range(0, len(temp_dataset), batch_size)):
        i_trial = 0
        while i_trial < max_trials:
            try:
                result = evaluate(
                    dataset=temp_dataset.select(range(i, i + batch_size)),
                    metrics=[
                        context_precision,
                        context_recall,
                        context_relevancy
                    ],
                )
                break
            except Exception as e:
                print(f"error in evaluation: {e}")
                sleep(5)
                i_trial += 1
        else:
            print("failed to evaluate")
            result = {key: None for key in scores.keys()}

        for key in scores.keys():
            scores[key].append(result[key])

    return scores


def generator_llm_evaluate(dataset, model_name):
    model = OpenAI(model_name)
    faithfulness.llm = model
    faithfulness.batch_size = 5
    result = evaluate(
        dataset=dataset.map(lambda example: {'ground_truths': [example['ground_truth']]}),
        metrics=[
            faithfulness,
        ],
    )
    return result


def full_pipeline_llm_evaluate(dataset, model_name):
    model = OpenAI(model_name)
    answer_correctness.llm = model
    answer_correctness.batch_size = 15
    answer_correctness.weights = [0.9, 0.1]
    answer_correctness.answer_similarity = AnswerSimilarity(llm=model, batch_size=15)
    result = evaluate(
        dataset=dataset.map(lambda example: {'ground_truths': [example['ground_truth']]}),
        metrics=[
            answer_correctness,
        ],
    )
    return result
