# Usage Guide for LM Evaluation Harness with Financial Phrasebank

This guide provides instructions on how to use the `lm-evaluation-harness` library with added F1 metric to evaluate Agents or Models on the datasets using various prompts.

## Setup

1. **Clone the Repository**
   ```
   git clone https://github.com/EleutherAI/lm-evaluation-harness
   cd lm-evaluation-harness
   ```

2. **Install Dependencies**
   ```
   pip install -e .
   ```

3. **As an example add Financial Phrasebank Dataset** 
   Ensure the Financial Phrasebank dataset is available in the required format. You can download it from [Financial Phrasebank](https://huggingface.co/datasets/financial_phrasebank/tree/main).

4. **Add Prompt Templates**
   Use a file named `prompt_templates.txt` in the repository directory. This file should contain different prompt templates to be used for zero-shot classification tasks.

## Evaluation



## Additional Notes

- Ensure the dataset and prompt templates are correctly formatted.
- Refer to the [LM Evaluation Harness documentation](https://github.com/EleutherAI/lm-evaluation-harness) for detailed usage instructions and troubleshooting.
