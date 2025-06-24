"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --model_size 7b --behaviors sycophancy
"""

import json
import torch as t
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class ComparisonDataset(Dataset):
    def __init__(self, data_path, token, model_name_path, use_chat, behavior):
        # with open(data_path, "r") as f:
        #     self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        tokenized_demonstration_list = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
            data=data.split('\n\n\n')
            if data[-1] == '':
                data.pop()
            # data_all = data[:32]
            if behavior=="gsm8k" or behavior=="multia":
                data_all = data[:20]
            elif behavior=="addsub" or behavior=="asdiv" or behavior=="svamp"or behavior=="mawps":
                data_all = data[:40]
            quest_all = []
            answer_all = []
            for text in data_all:
                q_part, a_part = text.split("Let's think step by step.")
                a_part = "A: Let's think step by step." + a_part
                quest_all.append(q_part)
                answer_all.append(a_part)
            self.data = []
            for i in range(len(quest_all)):
                j = random.choice([k for k in range(len(answer_all)) if k != i])
                item = {"answer_matching_behavior":answer_all[i],
                        "answer_not_matching_behavior":answer_all[j],
                        "question":quest_all[i]}
                self.data.append(item)

    def prompt_to_tokens(self, instruction, model_output):
        if self.use_chat:  # F
            tokens = tokenize_llama_chat(  # chat 模式会包上 [INST] prompt [/INST]，符合 Llama Chat 格式
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        else:
            tokens = tokenize_llama_base(  # base 模式就是直接拼接 question + answer
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]  # positive completion
        n_text = item["answer_not_matching_behavior"]  # negative completion
        q_text = item["question"]  # prompt 
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens

def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: List[str],
    model: LlamaWrapper,
    data_path,
):
    # data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    model.set_save_internal_decodings(False)  # 用于控制是否保存中间解码结果的开关
    model.reset_all()  # 状态清理的核心机制，它确保每次前向传播时，模型不会“残留上一次的行为 steer 信息

    pos_activations = dict([(layer, []) for layer in layers])  # 生成 steering vector（行为引导向量） 的前置步骤，用于为每一层准备存储容器
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
        behavior,
    )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()  # 清空模型中每一层记录的激活值、dot product 追踪等状态。确保每次前向传播干净、不会混入上一个样本的中间状态。
        model.get_logits(p_tokens)  # 对当前正样本 token（p_tokens）进行前向传播，并在每一层的 BlockOutputWrapper 中保存当前 residual stream 的激活。
        for layer in layers:
            p_activations = model.get_last_activations(layer)  # 因为 tokenizer 的最后一个 token 是 eos_token（例如 <\s>），不代表语义，是 padding/终止标记。
            p_activations = p_activations[0, -2, :].detach().cpu() # 0：batch size = 1   -2：最后一个 token 的前一个位置  :：hidden size，全维 residual stream 向量
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer]) # 每一层都有1000个样本，每个样本是4096维  pos_activations是32✖️1000✖️4096
        all_neg_layer = t.stack(neg_activations[layer])  # all_pos_layer.shape [1000, 4096]
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)  # torch.Size([4096]) 计算 mean difference vector（steering vector）
        t.save( 
            vec,
            get_vector_path(behavior, layer, model.model_name_path),
        )  # 保存向量 & 激活
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )

def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
    model: str,
    data_path,
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    behaviors: behaviors to generate vectors for
    """
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model, model_name_path=model
    )
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model,data_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="7b")

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors,
        args.model,
        args.data_path
    )
