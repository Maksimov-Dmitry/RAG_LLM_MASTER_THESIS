import hydra
from omegaconf import DictConfig
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel
from datasets import load_from_disk
from datasets.filesystems import S3FileSystem
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from src.entities.train_generator_params import TrainGeneratorParams, read_train_generator_params
import subprocess

import logging
from sagemaker.remote_function import RemoteExecutor
import sagemaker
from dataclasses import asdict
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ds_preprocessing(row):
    for i, doc in enumerate(row['contexts_documents']):
        if doc == row['document'] and row['page'] == row['contexts_pages'][i]:
            row['ground_truth'] = row['ground_truth']
            return row
    row['ground_truth'] = 'Leider kann ich diese Frage nicht einmal mit den bereitgestellten Informationen beantworten'
    return row


def formatting_func(example):
    system_prompt = "Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer.\nDer Assistent gibt ausführliche, hilfreiche und ehrliche Antworten."
    background = '\n\n'
    for i, (doc_name, document) in enumerate(zip(example['contexts_documents'], example['contexts']), 1):
        background += f"Dokument {i}: {doc_name}.\n{document}\n\n"
    prompt = example['question']
    answer = example['ground_truth']
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nBeantworten Sie die folgende Frage basierend auf dem Kontext.\nKontext: {background}\nFrage: {prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    return text


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['ground_truth'])):
        system_prompt = "Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer.\nDer Assistent gibt ausführliche, hilfreiche und ehrliche Antworten."
        background = '\n\n'
        for j, (doc_name, document) in enumerate(zip(example['contexts_documents'][i], example['contexts'][i]), 1):
            background += f"Dokument {j}: {doc_name}.\n{document}\n\n"
        prompt = example['question'][i]
        answer = example['ground_truth'][i]
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nBeantworten Sie die folgende Frage basierend auf dem Kontext.\nKontext: {background}\nFrage: {prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        output_texts.append(text)
    return output_texts


def train_generator(params: TrainGeneratorParams):
    logger.info(f"start train generator with params {params}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        params.model_input_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        bias="none",
        task_type="CAUSAL_LM",
        **asdict(params.lora_config)
    )

    tokenizer = AutoTokenizer.from_pretrained(params.model_input_name, padding_side='right')
    s3 = S3FileSystem()
    dataset = load_from_disk(f's3://{params.bucket_name}/{params.input_data}', storage_options=s3.storage_options)
    dataset = dataset.map(ds_preprocessing)
    if params.use_DataCollatorForCompletionOnlyLM:
        response_template = "assistant\n"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir='results',
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1 / (10 * params.training_arguments.num_train_epochs),
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        load_best_model_at_end=True,
        **asdict(params.training_arguments)
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        peft_config=peft_config,
        max_seq_length=params.max_seq_length,
        packing=not params.use_DataCollatorForCompletionOnlyLM,
        formatting_func=formatting_prompts_func if params.use_DataCollatorForCompletionOnlyLM else formatting_func,
        data_collator=collator if params.use_DataCollatorForCompletionOnlyLM else None,
    )

    trainer.train()

    base_model = AutoModelForCausalLM.from_pretrained(
        params.model_input_name,
        return_dict=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(params.model_input_name)

    model = PeftModel.from_pretrained(base_model, trainer.state.best_model_checkpoint)
    model = model.merge_and_unload()
    model.save_pretrained('new_model')
    tokenizer.save_pretrained('new_model')

    command = [
        "aws", "s3", "cp",
        "new_model",
        f"s3://{params.bucket_name}/{params.model_output_name}",
        "--recursive"
    ]

    subprocess.run(command)


@hydra.main(version_base=None, config_path="../configs", config_name="train_generator_config")
def train_generator_command(cfg: DictConfig):
    params = read_train_generator_params(cfg)
    load_dotenv()
    sm_session = sagemaker.Session(
        boto_session=boto3.session.Session(region_name="eu-central-1"),
        default_bucket='tcr-algotrading'
    )
    with RemoteExecutor(
        sagemaker_session=sm_session,
        volume_size=50,
        instance_type=params.sagemaker_instance,
        instance_count=1,
        environment_variables={'WANDB_PROJECT': os.getenv('WANDB_PROJECT'), 'WANDB_API_KEY': os.getenv('WANDB_API_KEY')},
        image_uri='763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker',
        job_name_prefix='generator-training',
        dependencies='sagemaker_train_generator_requirements.txt',
        include_local_workdir=True,
        role='arn:aws:iam::185705041424:role/SageMakerRole',
    ) as e:
        future = e.submit(train_generator, params)
        print(future.result())


if __name__ == "__main__":
    train_generator_command()
