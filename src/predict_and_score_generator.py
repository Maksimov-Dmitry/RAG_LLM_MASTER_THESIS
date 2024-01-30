import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import load_from_disk, concatenate_datasets
from datasets.filesystems import S3FileSystem
from aim import Run
from dataclasses import asdict

from src.entities.predict_and_score_generator_config import PredictGeneratorParams, read_predict_generator_params
from src.evaluation.evaluation import generator_llm_evaluate
import logging
from sagemaker.remote_function import RemoteExecutor
import sagemaker
from tqdm import tqdm
import boto3
from dotenv import load_dotenv
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

httpx_logger = logging.getLogger('httpx')
openai_logger = logging.getLogger('openai._base_client')

httpx_logger.setLevel(logging.ERROR)
openai_logger.setLevel(logging.ERROR)


def predict_generator(params: PredictGeneratorParams):
    logger.info(f"start predict generator with params {params}")

    if 'dmitrii' in params.model:
        command = [
            "aws", "s3", "cp",
            f"s3://{params.bucket_name}/{params.model}",
            params.model,
            "--recursive"
        ]
        subprocess.run(command)
    if 'Mixtral' in params.model:
        model = AutoModelForCausalLM.from_pretrained(params.model, device_map="auto", revision='gptq-4bit-32g-actorder_True')
    else:
        model = AutoModelForCausalLM.from_pretrained(params.model, device_map="auto", torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2" if params.use_flash_attention_2 else None)

    tokenizer = AutoTokenizer.from_pretrained(params.model)

    s3 = S3FileSystem()
    dataset = load_from_disk(f's3://{params.bucket_name}/{params.input}', storage_options=s3.storage_options)

    if 'Mixtral' in params.model:
        if params.use_context:
            prompt_format = "Beantworten Sie die folgende Frage basierend auf dem Kontext.\nKontext: {background}\nFrage: {prompt}."
        else:
            prompt_format = "Beantworten Sie die folgende Frage.\nFrage: {prompt}."
    else:
        system_prompt = "Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer.\nDer Assistent gibt ausführliche, hilfreiche und ehrliche Antworten."
        if params.use_context:
            prompt_format = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nBeantworten Sie die folgende Frage basierend auf dem Kontext.\nKontext: {background}\nFrage: {prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt_format = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nBeantworten Sie die folgende Frage.\nFrage: {prompt}<|im_end|>\n<|im_start|>assistant\n"

    generator = pipeline(model=model, tokenizer=tokenizer, task='text-generation', return_full_text=False)
    stages = ['val', 'test']
    if 'mistral' in params.model:
        stages += ['train']
    for stage in stages:
        results = []
        for row in tqdm(dataset[stage]):
            background = '\n\n'
            for i, (doc_name, document) in enumerate(zip(row['contexts_documents'], row['contexts']), 1):
                background += f"Dokument {i}: {doc_name}.\n{document}\n\n"
            if 'Mixtral' in params.model:
                try:
                    if params.use_context:
                        inp = prompt_format.format(prompt=row['question'], background=background)
                    else:
                        inp = prompt_format.format(prompt=row['question'])
                    inp = generator.tokenizer.apply_chat_template([{"role": "user", "content": inp}], tokenize=False, add_generation_prompt=True)
                    res = generator(inp, do_sample=True, top_p=params.top_p, max_new_tokens=params.max_new_tokens,
                                    temperature=params.temperature, top_k=50)
                except RuntimeError as e:
                    res = [{'generated_text': ''}]
                    logger.error(f"RuntimeError: {e}")
            else:
                if params.use_context:
                    inp = prompt_format.format(prompt=row['question'], system_prompt=system_prompt, background=background)
                else:
                    inp = prompt_format.format(prompt=row['question'], system_prompt=system_prompt)
                res = generator(inp, do_sample=True, top_p=params.top_p, max_new_tokens=params.max_new_tokens,
                                temperature=params.temperature)
            results.append(res[0]['generated_text'])
        dataset[stage] = dataset[stage].add_column('answer', results)
    dataset.save_to_disk(f's3://{params.bucket_name}/{params.output}', storage_options=s3.storage_options)


@hydra.main(version_base=None, config_path="../configs", config_name="predict_generator_config")
def predict_generator_command(cfg: DictConfig):
    params = read_predict_generator_params(cfg)
    if params.use_flash_attention_2:
        pre_execution_commands = ['pip install flash-attn==2.4.2']
    elif 'GPTQ' in params.model:
        pre_execution_commands = ['pip install -U torch==2.1.0 optimum',
                                  'pip install -U git+https://github.com/huggingface/transformers.git',
                                  'pip install -U auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/']
    elif 'AWQ' in params.model:
        pre_execution_commands = ['pip install -U torch==2.1.0 autoawq==0.1.8']
    else:
        pre_execution_commands = None
    sm_session = sagemaker.Session(
        boto_session=boto3.session.Session(region_name="eu-central-1"),
        default_bucket='tcr-algotrading'
    )
    with RemoteExecutor(
        sagemaker_session=sm_session,
        instance_type=params.sagemaker_instance,
        instance_count=1,
        image_uri='763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker',
        job_name_prefix='generator-predict',
        include_local_workdir=True,
        pre_execution_commands=pre_execution_commands,
        dependencies='sagemaker_predict_generator_requirements.txt',
        role='arn:aws:iam::185705041424:role/SageMakerRole',
    ) as e:
        future = e.submit(predict_generator, params)
        print(future.result())
    if params.use_context:
        load_dotenv()
        s3 = S3FileSystem()
        dataset = load_from_disk(f's3://{params.bucket_name}/{params.output}', storage_options=s3.storage_options)
        run = Run(experiment='predict_generator', capture_terminal_logs=False)
        run['hparams'] = asdict(params)
        for stage in ['val', 'test']:
            scores = generator_llm_evaluate(dataset[stage], params.evaluator_model)
            dataset[stage] = concatenate_datasets([dataset[stage], scores.scores], axis=1)
            for key, value in scores.items():
                run.track(value, name=key, context={'subset': stage})
        dataset.save_to_disk(f's3://{params.bucket_name}/{params.output}', storage_options=s3.storage_options)


if __name__ == "__main__":
    predict_generator_command()
