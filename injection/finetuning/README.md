## Finetuning LLaMA 2 with Petals Distributed Model

This repository contains two scripts for fine-tuning the LLaMA 2 model. The first script uses the standard fine-tuning approach, and the second script uses the Petals distributed model for more efficient fine-tuning with prompt-tuning.

### Method 1: Standard Fine-tuning

This method uses the `transformers` library to load and fine-tune the LLaMA 2 model on a given dataset. The dataset is tokenized and then used to train the model with all layers frozen except for additional learnable embeddings.

#### Script Overview:

- **Tokenization**: The dataset is tokenized using the `LlamaTokenizer`.
- **Model Preparation**: The LLaMA 2 model is loaded with frozen layers except for the added embeddings.
- **Training**: The model is trained on the tokenized dataset with specified hyperparameters. The model is saved every 20,000 steps.
- **Logging**: Training loss is logged using Weights & Biases.

### Method 2: Fine-tuning with Petals Distributed Model

This method leverages the Petals library to fine-tune the LLaMA 2 model using a distributed approach. This allows for efficient prompt-tuning, where separate prefixes are fine-tuned for each transformer block.

#### Script Overview:

- **Tokenization**: The dataset is tokenized using the `LlamaTokenizer`.
- **Model Preparation**: The LLaMA 2 model is loaded using the Petals `DistributedLlamaForCausalLM` class with prompt-tuning.
- **Training**: The model is trained on the tokenized dataset with specified hyperparameters. The model is saved every N steps.

### Key Hyperparameters:

- **NUM_PREFIX_TOKENS**: Number of prefix tokens used in prompt-tuning.
- **DEVICE**: The device used for training (e.g., 'cuda').
- **BATCH_SIZE**: The batch size for training.
- **LR**: Learning rate for the optimizer.
- **WEIGHT_DECAY**: Weight decay for the optimizer.
- **NUM_SAMPLES**: Number of samples from the dataset to use for training.
- **SEED**: Random seed for reproducibility.
- **MODEL_MAX_LENGTH**: Maximum sequence length for the tokenizer.
