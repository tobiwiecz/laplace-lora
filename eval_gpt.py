import argparse
import json
import logging
import os

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    default_data_collator,
)
from peft import PeftModel
from metrics import compute_all_metrics

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc evaluation of a fine-tuned LoRA checkpoint")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to a step_N directory containing adapter_config.json and adapter_model.safetensors.",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["val"],
        choices=["train", "val", "test"],
        help="Which dataset splits to evaluate on.",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=400)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write results. Defaults to checkpoint_dir.",
    )
    parser.add_argument("--seed_label", type=str, default=None,
        help="Human-readable seed label (e.g. seed1) for results/ path.")
    parser.add_argument("--results_dir", type=str, default=None,
        help="If set, also writes results/{task}/{seed_label}/mean.json here.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir

    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load raw dataset
    if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
        raw_datasets = load_dataset("glue", args.task_name)
    elif args.task_name in ['cb', 'wic', 'boolq']:
        raw_datasets = load_dataset("super_glue", args.task_name)
    elif 'ARC' in args.task_name:
        raw_datasets = load_dataset('ai2_arc', args.task_name)
    elif 'winogrande' in args.task_name:
        raw_datasets = load_dataset('winogrande', args.task_name)
    else:
        raw_datasets = load_dataset(args.task_name)

    if 'ARC' in args.task_name or 'openbookqa' in args.task_name:
        for split in list(raw_datasets.keys()):
            raw_datasets[split] = raw_datasets[split].filter(
                lambda ex: len(ex['choices']['label']) == 4
            )

        def convert_choices_to_alpha(example):
            mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
            example['choices']['label'] = [mapping.get(l, l) for l in example['choices']['label']]
            example['answerKey'] = mapping.get(example['answerKey'], example['answerKey'])
            example['choices']['text'] = [t if t.endswith('.') else t + '.' for t in example['choices']['text']]
            example['choices']['text'] = [t[0].upper() + t[1:] if t else t for t in example['choices']['text']]
            return example

        for split in list(raw_datasets.keys()):
            raw_datasets[split] = raw_datasets[split].map(convert_choices_to_alpha)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer,
        padding_side='left', token=True,
    )
    tokenizer.pad_token = tokenizer.bos_token
    if args.task_name in ['boolq']:
        tokenizer.add_eos_token = True

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        token=True,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model.eval()

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        if args.task_name == 'boolq':
            texts = [
                f"Answer the question with only True or False: {q} Context: {p}"
                for p, q in zip(examples['passage'], examples['question'])
            ]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        elif 'openbookqa' in args.task_name:
            choices_list = [
                ' '.join(f'{l}. {t}' for l, t in zip(c['label'], c['text']))
                for c in examples['choices']
            ]
            texts = [
                f"Select one of the choices that answers the following question: {q} Choices: {c} Answer:"
                for q, c in zip(examples['question_stem'], choices_list)
            ]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[l] for l in examples["answerKey"]]
        elif 'ARC' in args.task_name:
            choices_list = [
                ' '.join(f'{l}. {t}' for l, t in zip(c['label'], c['text']))
                for c in examples['choices']
            ]
            texts = [
                f"Select one of the choices that answers the following question: {q} Choices: {c} Answer:"
                for q, c in zip(examples['question'], choices_list)
            ]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[l] for l in examples["answerKey"]]
        elif 'winogrande' in args.task_name:
            texts = [
                f"Select one of the choices that answers the following question: {q} Choices: A. {o1}. B {o2}. Answer:"
                for q, o1, o2 in zip(examples['sentence'], examples['option1'], examples['option2'])
            ]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"1": 0, "2": 1, "": None}
            result["labels"] = [map_dict[l] for l in examples["answer"]]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            if args.task_name == 'boolq':
                self.id_list = [tokenizer.encode('False', add_special_tokens=False)[0], tokenizer.encode('True', add_special_tokens=False)[0]]
            elif args.task_name in ['openbookqa'] or 'ARC' in args.task_name:
                self.id_list = [
                    tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0],
                    tokenizer.encode('C', add_special_tokens=False)[0], tokenizer.encode('D', add_special_tokens=False)[0],
                ]
            elif 'winogrande' in args.task_name:
                self.id_list = [tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0]]
            self.model = model

        def forward(self, **kwargs):
            kwargs.pop('labels', None)
            output_dict = self.model(**kwargs)
            output_dict['logits'] = output_dict['logits'][:, -1, self.id_list]
            return output_dict

    model = WrappedModel(model)
    model = accelerator.prepare(model)

    data_collator = (
        default_data_collator if args.pad_to_max_length
        else DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None),
        )
    )

    split_to_hf = {
        'train': 'train',
        'val': 'validation_matched' if args.task_name == 'mnli' else 'validation',
        'test': 'test',
    }

    for split_name in args.splits:
        hf_split = split_to_hf[split_name]
        if hf_split not in processed_datasets:
            logger.warning(f"Split '{hf_split}' not available for {args.task_name}, skipping.")
            continue

        # Drop unlabeled examples (e.g. winogrande test has answer="", boolq test has label=-1)
        dataset = processed_datasets[hf_split].filter(
            lambda ex: ex['labels'] is not None and ex['labels'] != -1
        )
        if len(dataset) == 0:
            logger.warning(f"Split '{split_name}' has no labeled examples, skipping.")
            continue

        logger.info(f"Evaluating {split_name} split: {len(dataset)} examples")

        dataloader = DataLoader(
            dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        dataloader = accelerator.prepare(dataloader)

        _exp_id = (args.checkpoint_dir + split_name).replace('/', '_').replace('.', '').strip('_')
        if args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
            metric = evaluate.load("glue", args.task_name, experiment_id=_exp_id)
        elif args.task_name in ['cb', 'wic', 'boolq']:
            metric = evaluate.load("super_glue", args.task_name, experiment_id=_exp_id)
        else:
            metric = evaluate.load('accuracy', experiment_id=_exp_id)

        model.eval()
        samples_seen = 0
        all_logits = []
        all_labels = []

        for step, batch in tqdm(enumerate(dataloader), desc=f"Evaluating {split_name}", total=len(dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            logits = outputs.logits.detach()

            predictions, references, logits = accelerator.gather(
                (predictions, batch["labels"], logits)
            )
            if accelerator.num_processes > 1:
                if step == len(dataloader) - 1:
                    n_keep = len(dataloader.dataset) - samples_seen
                    predictions = predictions[:n_keep]
                    references = references[:n_keep]
                    logits = logits[:n_keep]
                else:
                    samples_seen += references.shape[0]

            all_logits.append(logits.cpu())
            all_labels.append(references.cpu())
            metric.add_batch(predictions=predictions, references=references)

        eval_metric = metric.compute()
        logger.info(f"{split_name}: {eval_metric}")

        if accelerator.is_main_process:
            all_logits_cat = torch.cat(all_logits, dim=0)
            all_labels_cat = torch.cat(all_labels, dim=0)
            all_probs_cat = torch.softmax(all_logits_cat, dim=-1)

            all_results = {k.removeprefix("eval_"): v for k, v in eval_metric.items()}
            all_results.update(compute_all_metrics(all_probs_cat, all_labels_cat))

            all_results_path = os.path.join(args.output_dir, f"all_results_{split_name}.json")
            if os.path.isfile(all_results_path):
                os.remove(all_results_path)
            with open(all_results_path, "w") as f:
                json.dump(all_results, f)

            output_dicts = [
                {
                    'index': j,
                    'true': all_labels_cat[j].item(),
                    'pred': all_logits_cat[j].argmax().item(),
                    'conf': all_probs_cat[j].max().item(),
                    'logits': all_logits_cat[j].numpy().tolist(),
                    'probs': all_probs_cat[j].numpy().tolist(),
                }
                for j in range(all_logits_cat.size(0))
            ]
            output_path = os.path.join(args.output_dir, f"eval_res_{split_name}.json")
            print(f"writing outputs to '{output_path}'")
            if os.path.isfile(output_path):
                os.remove(output_path)
            with open(output_path, 'w+') as f:
                for output_dict in output_dicts:
                    f.write(f'{json.dumps(output_dict)}\n')

            if args.results_dir and args.seed_label:
                suffix = "" if split_name == "val" else f"_{split_name}"
                clean_dir = os.path.join(args.results_dir, args.task_name, args.seed_label)
                os.makedirs(clean_dir, exist_ok=True)
                clean_path = os.path.join(clean_dir, f"mean{suffix}.json")
                with open(clean_path, "w") as f:
                    json.dump({k.removeprefix("eval_"): v for k, v in all_results.items()}, f, indent=2)
                print(f"Results saved → {clean_path}")


if __name__ == "__main__":
    main()
