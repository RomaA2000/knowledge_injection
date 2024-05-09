import argparse
from cloudpathlib import CloudPath, S3Client
from pathlib import Path
import transformers
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments as HfTrainingArguments,
)
import torch
from torch.nn import Embedding
from datasets import load_dataset
import os
from dataclasses import dataclass, field
from typing import Optional

LOCAL_SOURCE_MODEL_DIR = "/assets/model"
LOCAL_DST_MODEL_DIR = "/assets/model_result"
LOCAL_JSONL_DATASET_FILE = "/assets/jsonl/fpb.jsonl"
LOCAL_TRAINING_LOGS_DIR = "/assets/logs"
LOCAL_CKPTS_DIR = "/assets/ckpts"
NEW_TOKEN_SIZE = 256

LOCAL_DIRS = [
    LOCAL_SOURCE_MODEL_DIR, 
    LOCAL_DST_MODEL_DIR, 
    LOCAL_TRAINING_LOGS_DIR,
    LOCAL_CKPTS_DIR,
    str(Path(LOCAL_JSONL_DATASET_FILE).parent)
]

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

@dataclass
class TrainingArguments(HfTrainingArguments):
    base_model_s3_dir: Optional[str] = field(default=None)
    jsonl_dataset_s3_file: Optional[str] = field(default=None)
    saving_model_s3_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=LOCAL_CKPTS_DIR)
    gradient_checkpointing: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    evaluation_strategy: Optional[str] = field(default="no")
    remove_unused_columns: Optional[bool] = field(default=False)
    logging_dir: Optional[str] = field(default=LOCAL_TRAINING_LOGS_DIR)
    logging_strategy: Optional[str] = field(default="steps")
    logging_steps: Optional[int] = field(default=50)
    save_strategy: Optional[str] = field(default="epoch")
    save_total_limit: Optional[int] = field(default=1)
    disable_tqdm: Optional[bool] = field(default=False)
    push_to_hub: Optional[bool] = field(default=False)
    save_steps: Optional[int] = field(default=20000)
    per_device_train_batch_size: Optional[int] = field(default=64)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    bf16: Optional[bool] = field(default=True)
    learning_rate: Optional[float] = field(default=5e-5)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    weight_decay: Optional[float] = field(default=0.0)
    warmup_ratio: Optional[float] = field(default=0.03)
    num_gpus: Optional[int] = field(default=1)

def create_local_dirs_if_not_exist():
    for local_dir in LOCAL_DIRS:
        local_dir_path = Path(local_dir)
        if not local_dir_path.exists():
            local_dir_path.mkdir(parents=True, exist_ok=True)

def download_assets(args):
    s3_client = S3Client(
        aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    
    model_cloud_path = CloudPath(cloud_path=args.base_model_s3_dir, client=s3_client)
    model_cloud_path.download_to(LOCAL_SOURCE_MODEL_DIR)

    dataset_cloud_path = CloudPath(cloud_path=args.jsonl_dataset_s3_file, client=s3_client)
    dataset_cloud_path.download_to(LOCAL_JSONL_DATASET_FILE)

def upload_model(args):
    s3_client = S3Client(
        aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    
    model_cloud_path = CloudPath(cloud_path=args.saving_model_s3_dir, client=s3_client)
    model_cloud_path.upload_from(LOCAL_DST_MODEL_DIR, force_overwrite_to_cloud=True)

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, add_embeddings_size=NEW_TOKEN_SIZE):
        super().__init__(config)
        self.add_embeddings_size = add_embeddings_size
        self.additional_embeddings = Embedding(config.max_position_embeddings, add_embeddings_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        original_inputs_embeds = self.model.embed_tokens(input_ids)
        
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        add_embeds = self.additional_embeddings(position_ids)

        inputs_embeds = torch.cat((add_embeds, original_inputs_embeds), dim=1)
        
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


def do_training(args):
    def preprocessing_fn(examples):
        return tokenizer(examples["text"], return_tensors="pt", padding=True)
    
    tokenizer = LlamaTokenizer.from_pretrained(LOCAL_SOURCE_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("json", data_files=LOCAL_JSONL_DATASET_FILE)
    tokenized_dataset = dataset.map(
        preprocessing_fn,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    
    model = CustomLlamaForCausalLM.from_pretrained(
        LOCAL_SOURCE_MODEL_DIR,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )

    for param in model.parameters():
        param.requires_grad = False
    
    model.additional_embeddings.weight.requires_grad = True
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"]
    )

    trainer.train()
    
    trainer.save_model(LOCAL_DST_MODEL_DIR)
    tokenizer.save_pretrained(LOCAL_DST_MODEL_DIR)

def main():
    parser = transformers.HfArgumentParser((TrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    create_local_dirs_if_not_exist()
    download_assets(args)
    do_training(args)
    upload_model(args)

if __name__ == "__main__":
    main()
