import os

import torch
import transformers
import wandb
from datasets import load_dataset
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, get_scheduler

from petals import DistributedLlamaForCausalLM

MODEL_NAME = "testing/llama-2-7b"
dataset = load_dataset("fpb")
TUNING_MODE = 'ptune'
NUM_PREFIX_TOKENS = 256
DEVICE = 'cuda'
BATCH_SIZE = 1
LR = 5e-5
WEIGHT_DECAY = 0.0
SEED = 42
MODEL_MAX_LENGTH = 1024 + NUM_PREFIX_TOKENS

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'right'
tokenizer.model_max_length = MODEL_MAX_LENGTH

model = DistributedLlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    pre_seq_len=NUM_PREFIX_TOKENS,
    tuning_mode=TUNING_MODE
).to(DEVICE)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MODEL_MAX_LENGTH)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    drop_last=True,
)

for n, p in model.named_parameters():
    if p.requires_grad:
        print(n, p.requires_grad, p.device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)
)


for step, batch in enumerate(tqdm(train_dataloader)):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    print("Train Loss:", loss.item())

    if step > 0 and step % 5000 == 0:
        model.save_pretrained(f"checkpoint-{step}")
        tokenizer.save_pretrained(f"checkpoint-{step}")

model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")