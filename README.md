# Fine-Tuning & Retrieval-Augmented Generation with Mistral-7B using the Dolly dataset 

This project explores the fine-tuning of Mistral-7B on the [Dolly 15k dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)  and integrates a Retrieval-Augmented Generation (RAG) pipeline. The work systematically evaluates how fine-tuning and retrieval affect model performance on natural language generation tasks.

## Overview 

The project investigates four experimental configurations:
- Base Mistral-7B
- Base Mistral-7B + RAG
- Fine-tuned Mistral-7B (on Dolly 15k)
- Fine-tuned Mistral-7B + RAG

The evaluation is done using the **_F1 Score_** for token-level accurancy and **_BERTScore_** to capture semantic similarities between the predictions and the truth values. 


## How to run 

```bash
## First install the code
git clone git@github.com:jmsardain/dolly.git 
cd dolly

## Run the code 
##### To train 
python main.py --train
##### To evaluate all 4 configurations 
python main.py --evaluate
```

## Results and observations

| Configuration            | F1 Score | BERTScore |
| ------------------------ | -------- | --------- |
| Base model, no RAG       |  0.1379  | 0.8158 |
| Base model, RAG          |  0.1292  | 0.8181 |
| Fine-tuned model, no RAG |  0.1423  | 0.8155 |
| Fine-tuned model, RAG    |  0.1341  | 0.8204 |

- Fine-tuning improves token-level accuracy (F1).
- RAG integration improves semantic similarity and factual grounding (BERTScore).
- Best trade-off: use fine-tuned model and RAG, which combining accuracy gain and increase contextual relevance. 

## Problems/Questions ?
For questions or contributions, please contact me and open an issue 🍻
