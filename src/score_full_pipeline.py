import click
from datasets import load_from_disk, concatenate_datasets
from datasets.filesystems import S3FileSystem
from dotenv import load_dotenv
from aim import Run
from src.evaluation.evaluation import full_pipeline_llm_evaluate
from rouge_score import rouge_scorer
from numpy import mean


def compute_rouge(example, scorer):
    reference = example['ground_truth']
    prediction = example['answer']

    scores = scorer.score(reference, prediction)
    for key, value in scores.items():
        example[f'{key}_recall'] = value.recall

    return example


@click.command()
@click.option('--bucket_name', default='tcr-internal', help='Bucket name.')
@click.option('--data', default='dmitrii/results/generator_predictions_mistral', help='Data path.')
@click.option('--evaluator_model', default='gpt-3.5-turbo-1106', help='Evaluator model.')
def score_full_pipeline(bucket_name, data, evaluator_model):
    load_dotenv()
    run = Run(experiment='score_full_pipeline', capture_terminal_logs=False)
    run['hparams'] = {'data': data}
    s3 = S3FileSystem()
    dataset = load_from_disk(f's3://{bucket_name}/{data}', storage_options=s3.storage_options)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for stage in ['val', 'test']:
        dataset[stage] = dataset[stage].map(lambda example: compute_rouge(example, scorer))
        for key in ['rouge1_recall', 'rouge2_recall', 'rougeL_recall']:
            run.track(mean(dataset[stage][key]), name=key, context={'subset': stage})
        scores = full_pipeline_llm_evaluate(dataset[stage], evaluator_model)
        dataset[stage] = concatenate_datasets([dataset[stage], scores.scores], axis=1)
        for key, value in scores.items():
            run.track(value, name=key, context={'subset': stage})
    dataset.save_to_disk(f's3://{bucket_name}/{data}', storage_options=s3.storage_options)


if __name__ == '__main__':
    score_full_pipeline()
