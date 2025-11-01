# BART Fine-Tuned for Dialogue Summarization
**Author:** Tony Wahab AKAKPO
**Date:** 2025-10-01

---

## Description
This project presents a fine-tuned BART model for summarizing dialogues in English. The original BART model was trained on news articles (CNN/Daily Mail) and has been adapted for the SamSum dataset to handle chat-style text.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model & Fine-Tuning](#model--fine-tuning)
- [Evaluation Results](#evaluation-results)
- [Usage](#usage)
- [Installation](#installation)
- [Hugging Face Model](#hugging-face-model)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project demonstrates how to fine-tune a **BART large model** for **summarization of dialogues** using the [SamSum dataset](https://huggingface.co/datasets/samsum). The goal is to generate **concise and informative summaries** of multi-turn dialogues, including those with abbreviations and informal language.

The workflow leverages **Hugging Face Transformers, Datasets, and Evaluate**, and was executed primarily in **Google Colab** using a GPU for efficiency.
---

## Dataset
The model was fine-tuned using the **SamSum dataset**, which consists of chat dialogues paired with human-written summaries.

**Dataset source:** [SamSum on Hugging Face](https://huggingface.co/datasets/knkarthick/samsum)

---
## Key Details
- **Base Model:** `facebook/bart-large-xsum`
- **Fine-Tuning Dataset:** `knkarthick/samsum`
- **Training Duration:** 1 hour 14 minutes on GPU
- **Fine-Tuned Model Repository:** [bcool315/bcool-bart-finetuned-dialogue](https://huggingface.co/bcool315/bcool-bart-finetuned-dialogue)
- **Programming Language:** Python 3
- **Libraries:** Transformers, Datasets, Evaluate, PyTorch, NLTK

---

## Project Structure

model_bart_finetuned-dialogue/
├── README.md # This file
├── .gitignore # Git ignore to skip large files
├── config.json # Model configuration
├── generation_config.json # Generation parameters
├── merges.txt # Tokenizer merges
├── special_tokens_map.json # Special tokens mapping
├── tokenizer_config.json # Tokenizer config
├── training_args.bin # Training arguments
├── vocab.json # Tokenizer vocabulary

> ⚠️ **Note:** `model.safetensors` is not included in GitHub due to its large size (~1.6GB). It is available for download via the Hugging Face repo link above.

---
## Preprocessing
- Cleaned dialogue texts using regex to remove unnecessary tags.
- Tokenized the text and summaries using `BartTokenizer`.
- Created attention masks and label IDs compatible with BART.
- Removed noise and irrelevant information to improve summarization quality.

---

## Model & Fine-Tuning
- **Base Model:** `facebook/bart-large-xsum`
- **Defined training arguments using:** `Seq2SeqTrainingArguments`
- **Fine-Tuning:** 4 epochs on the SamSum dataset
- **Batch size:** 4
- **Learning rate:** 2e-5
- **Device:** GPU recommended (Colab/Kaggle)
- **Library:** Hugging Face Transformers, Datasets, Evaluate

Fine-tuning allowed the model to better capture the structure and content of dialogues compared to the original news-trained BART.

---

## Evaluation Results
Performance on the **validation set**:

| Metric        | Score     |
|---------------|----------|
| Loss          | 1.4076     |
| ROUGE-1       | 53.51    |
| ROUGE-2       | 28.62    |
| ROUGE-L       | 44.15    |
| ROUGE-Lsum    | 49.14    |
| Avg. Gen Len  | 29.89 tokens |

- Summaries are concise and informative.
- The model handles abbreviations and chat-specific terms well.

---

## Deployment
    - Uploaded the fine-tuned model to **Hugging Face** for easy access:
      [bcool315/bcool-bart-finetuned-dialogue](https://huggingface.co/bcool315/bcool-bart-finetuned-dialogue)
    - Scripts and configuration files are maintained on **GitHub**.

---

## Usage
### 1 Using the Hugging Face Pipeline (simplest)

```python
from transformers import pipeline

# Load summarization pipeline with the fine-tuned model
summarizer = pipeline(
    "summarization",
    model="bcool315/bcool-bart-finetuned-dialogue"
)

text = """John: Hey! I've been thinking about getting a PlayStation 5. Do you think it is worth it?
Dan: Idk man. R u sure ur going to have enough free time to play it?
John: Yeah, that's why I'm not sure if I should buy one or not. I've been working so much lately idk if I'm gonna be able to play it as much as I'd like."""

generated_summary = summarizer(text)

print("Original Dialogue:\n")
print(text)
print("\nModel-generated Summary:\n")
print(generated_summary[0]['summary_text'])

```
### 2 Using Tokenizer and Model Directly (more flexible)
```python
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "bcool315/bcool-bart-finetuned-dialogue"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(**inputs, max_length=60, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Original Dialogue:\n")
print(text)
print("\nModel-generated Summary:\n")
print(summary)

```
 Tip: Use the pipeline for quick testing or demo purposes. Use the tokenizer + model approach when you need full control over generation parameters or batch processing. 

 ## Notes

- The model can handle abbreviations and informal language in dialogues.
- Large files like `model.safetensors` should be downloaded from **Hugging Face** to avoid GitHub size limitations.
- For further fine-tuning, you can reuse the configuration files and tokenizer included in this repo.
- Training on GPU in Google Colab took **1h14 min** for 4 epochs.

---

## License

This project is released under the **Apache 2.0 License**.  
See the [LICENSE](LICENSE) file for details.

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [SamSum Dataset](https://huggingface.co/datasets/samsum)
- Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*, 2019.
