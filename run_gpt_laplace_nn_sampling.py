import argparse
import json
import logging
import os
import random
import warnings
from pathlib import Path

from metrics import compute_all_metrics

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

import datasets
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: No GPU detected. This script requires a CUDA-capable GPU.")
from torch.nn.utils import parameters_to_vector, vector_to_parameters
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

from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PeftModel,
    PeftConfig
)

from laplace import Laplace
import pickle
import dill

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc NN (weight-space) sampling evaluation using a fitted Laplace approximation")
    parser.add_argument("--task_name", type=str, default='winogrande_s')
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--per_device_map_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument("--peft_method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--seed_label", type=str, default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true", default=True)
    parser.add_argument("--load_step", type=int, default=999)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--laplace_hessian", type=str, default='kron')
    parser.add_argument("--laplace_sub", type=str, default='last_layer')
    parser.add_argument("--laplace_prior", type=str, default='homo')
    parser.add_argument("--laplace_optim_step", type=int, default=1000)
    parser.add_argument("--testing_set", type=str, default='train_val')
    parser.add_argument("--lm_head", action="store_true", default=True)
    parser.add_argument("--n_samples", type=int, nargs='+', default=[1, 2, 4, 8],
                        help="List of MC sample counts, e.g. --n_samples 1 2 4 8. "
                             "max(list) forward passes are run; prefix averages give results for smaller counts.")
    parser.add_argument("--sampling_method", type=str, default='kron', choices=['kron', 'diag'],
                        help="'kron': full Kron posterior sampling respecting within-layer correlations. "
                             "'diag': diagonal marginal variances extracted from Kron posterior, independent per parameter.")
    args = parser.parse_args()

    peft_method = 'lora'
    if args.lm_head:
        peft_method = 'lora_lmhead'
    if args.testing_set != 'val':
        peft_method += args.testing_set

    seed_label = args.seed_label if args.seed_label is not None else args.seed
    args.output_dir += f'/{args.task_name}/{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{seed_label}'
    args.laplace_output_dir = f'outputs_laplace/{args.task_name}/{args.model_name_or_path}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{seed_label}/'

    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")

    return args


def diagonal_sample(la, n_samples):
    """Sample from the Laplace posterior using only the diagonal marginal variances.

    Extracts diag(P^{-1}) analytically from the KronDecomposed posterior precision.
    For each 2-factor layer: var[i,j] = Σ_{k1,k2} Q1[i,k1]² Q2[j,k2]² / (l1[k1]*l2[k2] + δ)
    Parameters are then sampled independently: θ ~ N(mean, diag(P^{-1})).
    Cheaper than full Kron sampling; loses within-layer correlations but retains
    the curvature-informed marginal uncertainties.
    """
    pp = la.posterior_precision
    diag_vars = []
    for ls, Qs, delta in zip(pp.eigenvalues, pp.eigenvectors, pp.deltas):
        if len(ls) == 1:
            Q, l = Qs[0], ls[0]
            # var[i] = Σ_k Q[i,k]² / (l[k] + δ)
            diag_vars.append((Q ** 2) @ (1.0 / (l + delta)))
        else:
            Q1, Q2 = Qs
            l1, l2 = ls
            # D[k1,k2] = 1 / (l1[k1]*l2[k2] + δ)
            D = 1.0 / (torch.ger(l1, l2) + delta)
            # var[i,j] = (Q1² @ D @ Q2²ᵀ)[i,j]
            diag_vars.append(((Q1 ** 2) @ D @ (Q2 ** 2).T).flatten())
    diag_std = torch.cat(diag_vars).sqrt()
    z = torch.randn(n_samples, la.n_params, device=la._device)
    return la.mean.unsqueeze(0) + z * diag_std.unsqueeze(0)


def main(load_step):
    args = parse_args()
    args.load_step = load_step
    laplace_output_dir = args.laplace_output_dir + f'step_{args.load_step}'
    os.makedirs(laplace_output_dir, exist_ok=True)

    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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
        filtered_train = raw_datasets["train"].filter(lambda example: len(example['choices']['label']) == 4)
        filtered_valid = raw_datasets["validation"].filter(lambda example: len(example['choices']['label']) == 4)
        filtered_test = raw_datasets["test"].filter(lambda example: len(example['choices']['label']) == 4)
        raw_datasets["train"] = filtered_train
        raw_datasets["validation"] = filtered_valid
        raw_datasets["test"] = filtered_test

        def convert_choices_to_alpha(example):
            mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
            example['choices']['label'] = [mapping.get(label, label) for label in example['choices']['label']]
            example['answerKey'] = mapping.get(example['answerKey'], example['answerKey'])
            example['choices']['text'] = [text if text.endswith('.') else text + '.' for text in example['choices']['text']]
            example['choices']['text'] = [text[0].upper() + text[1:] if text else text for text in example['choices']['text']]
            return example

        raw_datasets["train"] = raw_datasets["train"].map(convert_choices_to_alpha)
        raw_datasets["validation"] = raw_datasets["validation"].map(convert_choices_to_alpha)
        raw_datasets["test"] = raw_datasets["test"].map(convert_choices_to_alpha)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='left', token=True)
    tokenizer.pad_token = tokenizer.bos_token
    if args.task_name in ['boolq']:
        tokenizer.add_eos_token = True

    output_dir = args.output_dir + f'/step_{args.load_step}'

    peft_config = PeftConfig.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        dtype=torch.float16,
        token=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, output_dir)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'lora' in name:
            if 'all' in args.laplace_sub:
                param.requires_grad = True

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        if args.task_name == 'boolq':
            texts = [f"Answer the question with only True or False: {question} Context: {passage}" for passage, question in zip(examples['passage'], examples['question'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        elif 'openbookqa' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question_stem'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'ARC' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'winogrande' in args.task_name:
            texts = [f"Select one of the choices that answers the following question: {question} Choices: A. {option1}. B {option2}. Answer:" for question, option1, option2 in zip(examples['sentence'], examples['option1'], examples['option2'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"1": 0, "2": 1, "": None}
            result["labels"] = [map_dict[label] for label in examples["answer"]]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    processed_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    if args.testing_set == 'test':
        eval_dataset = processed_dataset.train_test_split(test_size=0.5, seed=42, shuffle=False)["test"]
    else:
        eval_dataset = processed_dataset

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if (accelerator.mixed_precision == "fp16") else None))

    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    map_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_map_batch_size)

    class CustomLMHead_lora(torch.nn.Module):
        def __init__(self, original_lm_head, id_list):
            super().__init__()
            self.id_list = id_list
            original_weight = original_lm_head.weight[id_list, :].clone()
            self.linear = torch.nn.Linear(in_features=original_weight.shape[1], out_features=len(id_list), bias=False).to(accelerator.device)
            self.linear.weight.data = original_weight.to(torch.float32)
            self.linear.weight.requires_grad = False
            self.lora_dropout = original_lm_head.lora_dropout['default']
            original_lora_A_weight = original_lm_head.lora_A["default"].weight.clone()
            self.lora_A = torch.nn.Linear(in_features=original_lora_A_weight.shape[1], out_features=original_lora_A_weight.shape[0], bias=False).to(accelerator.device)
            self.lora_A.weight.data = original_lora_A_weight.to(torch.float32)
            self.lora_A.weight.requires_grad = True
            original_lora_B_weight = original_lm_head.lora_B["default"].weight[id_list, :].clone()
            self.lora_B = torch.nn.Linear(in_features=original_lora_B_weight.shape[1], out_features=len(id_list), bias=False).to(accelerator.device)
            self.lora_B.weight.data = original_lora_B_weight.to(torch.float32)
            self.lora_B.weight.requires_grad = True
            self.scaling = args.lora_alpha / args.lora_r

        def forward(self, x):
            x_last = x[:, -1, :].to(torch.float32)
            result = self.linear(x_last)
            lora_result = self.lora_B(self.lora_A(self.lora_dropout(x_last)))
            return result + lora_result * self.scaling

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            if args.task_name == 'boolq':
                self.id_list = [tokenizer.encode('False', add_special_tokens=False)[0], tokenizer.encode('True', add_special_tokens=False)[0]]
            elif args.task_name == 'openbookqa':
                self.id_list = [tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0], tokenizer.encode('C', add_special_tokens=False)[0], tokenizer.encode('D', add_special_tokens=False)[0]]
            elif 'ARC' in args.task_name:
                self.id_list = [tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0], tokenizer.encode('C', add_special_tokens=False)[0], tokenizer.encode('D', add_special_tokens=False)[0]]
            elif 'winogrande' in args.task_name:
                self.id_list = [tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0]]

            if args.lm_head:
                original_lm_head = model.base_model.model.lm_head
                model.base_model.model.lm_head = CustomLMHead_lora(original_lm_head, self.id_list).to(accelerator.device)

            self.model = model

            if accelerator.is_main_process:
                model.print_trainable_parameters()
            accelerator.print(self.model)

        def forward(self, **kwargs):
            kwargs.pop('labels', None)
            output_dict = self.model(**kwargs)
            logits = output_dict['logits']
            if args.lm_head:
                selected_logits = logits
            else:
                selected_logits = logits[:, -1, self.id_list]
            return selected_logits.to(torch.float32)

    model = WrappedModel(model)

    model, eval_dataloader, map_dataloader = accelerator.prepare(model, eval_dataloader, map_dataloader)
    model.eval()

    la = Laplace(model, 'classification', prior_precision=1.,
                 subset_of_weights='all',
                 hessian_structure=args.laplace_hessian)

    # Load Hessian and prior precision saved by run_gpt_laplace.py
    H_path = f'{laplace_output_dir}/laplace_H_{args.laplace_hessian}_{args.laplace_sub}.pt'
    prior_precision_path = f'{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt'

    if not os.path.exists(H_path):
        raise FileNotFoundError(f'Laplace Hessian not found: {H_path}\nRun run_gpt_laplace.py first.')
    if not os.path.exists(prior_precision_path):
        raise FileNotFoundError(f'Prior precision not found: {prior_precision_path}\nRun run_gpt_laplace.py first.')

    H_state = torch.load(H_path, map_location=accelerator.device, weights_only=False)
    la.H = H_state['H']
    la.n_outputs = H_state['n_outputs']
    setattr(la.model, 'output_size', la.n_outputs)

    la.mean = parameters_to_vector(
        [p for n, p in model.named_parameters() if p.requires_grad and 'modules_to_save' not in n]
    ).detach()

    la.prior_precision = torch.load(prior_precision_path, map_location=accelerator.device)
    accelerator.print(f'Loaded Hessian from {H_path}')
    accelerator.print(f'Prior precision: {la.prior_precision}')

    # Parameters over which the Laplace posterior is defined (same filter as la.mean)
    trainable_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and 'modules_to_save' not in n]
    accelerator.print(f'Number of Laplace parameters: {la.n_params}')
    accelerator.print(f'Number of trainable params collected: {sum(p.numel() for p in trainable_params)}')

    n_samples_list = sorted(args.n_samples)
    max_samples = n_samples_list[-1]
    tag = f'nn_sampling_{args.sampling_method}_max{max_samples}_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}'
    accelerator.print(f'Sample counts: {n_samples_list}, running {max_samples} forward passes per example.')
    accelerator.print(f'Sampling method: {args.sampling_method}')

    def metrics_for(probs, labels):
        acc = (probs.argmax(-1) == labels).float().mean().item()
        m = compute_all_metrics(probs, labels)
        m['accuracy'] = acc
        return m

    # --- Pass 1: MAP (mean network) — fast, results printed immediately ---
    accelerator.print('\nRunning MAP evaluation...')
    all_map_probs = []
    all_labels = []

    for batch in tqdm(map_dataloader, total=len(map_dataloader), desc="Eval [MAP]", unit="batch"):
        input_batch = {k: v for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            map_logits = model(**input_batch)
            all_map_probs.append(torch.softmax(map_logits.float(), dim=-1).cpu())
        all_labels.append(batch["labels"].cpu())

    all_map_probs = torch.cat(all_map_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_results = {}
    all_results['mean'] = metrics_for(all_map_probs, all_labels)
    accelerator.print(f"  [mean] {all_results['mean']}")

    # --- Pass 2: weight-space sampling ---
    accelerator.print(f'\nRunning {args.sampling_method} sampling ({max_samples} max samples)...')
    all_sample_probs = []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval [{args.sampling_method}]", unit="batch"):
        input_batch = {k: v for k, v in batch.items() if k != 'labels'}

        with torch.no_grad():
            if args.sampling_method == 'kron':
                weight_samples = la.sample(max_samples)
            else:
                weight_samples = diagonal_sample(la, max_samples)

            batch_sample_probs = []
            for sample_w in weight_samples:
                vector_to_parameters(sample_w, trainable_params)
                logits = model(**input_batch)
                batch_sample_probs.append(torch.softmax(logits.float(), dim=-1).cpu())

            vector_to_parameters(la.mean, trainable_params)

        all_sample_probs.append(torch.stack(batch_sample_probs, dim=0))

    # (max_samples, n_test, n_classes)
    all_sample_probs = torch.cat(all_sample_probs, dim=1)

    for k in n_samples_list:
        avg_probs_k = all_sample_probs[:k].mean(0)
        key = f'{k}_sample' if k == 1 else f'{k}_samples'
        all_results[key] = metrics_for(avg_probs_k, all_labels)

    accelerator.print('Results:')
    for key, res in all_results.items():
        accelerator.print(f'  [{key}] {res}')

    all_results_path = os.path.join(output_dir, f'all_results_la_{tag}.json')
    if os.path.isfile(all_results_path):
        os.remove(all_results_path)
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    del model, la, all_sample_probs, all_map_probs, eval_dataloader, map_dataloader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args.load_step)
