from rag import retrieve
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from evaluate import load

def evaluate_model(model, dataset, index, tokenizer, model_RAG, instructions, use_rag=False, n=200):

    predictions = []
    truths = []

    bleu_scores = []
    f1_scores = []

    bert_scores = []
    metric = load("bertscore")

    for ex in dataset.select(range(n)):
        query = ex["instruction"]
        if use_rag:
            retrieved = retrieve(query, index, model_RAG, instructions, k=3)
            query = query + "\nRelevant info: " + " ".join(retrieved)
        gold = ex["response"]

        # pred = pipe(query, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
        pred = generate_with_model(query, model, tokenizer, max_new_tokens=100)
        pred_tokens = pred.split()
        gold_tokens = gold.split()
        # print(f"Gold: {gold}\n Pred: {pred}\n")
        truths.append(gold)
        predictions.append(pred)
        bleu_scores.append(sentence_bleu([gold_tokens], pred_tokens, smoothing_function=SmoothingFunction().method4)) ## use smooth function, BLEU isn't really great for paraphrasing but do it for now
        # crude F1: treat words as labels
        common = set(pred_tokens) & set(gold_tokens)
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        f1_scores.append(f1)

    ## now get BERT score from the predictions and truths
    bert_results = metric.compute(predictions=predictions, references=truths, lang="en")
    bertscore_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
    return sum(bleu_scores)/len(bleu_scores), sum(f1_scores)/len(f1_scores), bertscore_f1

def extract_answer(answer):

    answer_after = answer.split("### Response")[1]

    # if '###' in answer_after: ## in case of hallucination, the model predicts ### question again.
    #     answer_after = answer_after.split("###")[0]

    return answer_after

def generate_with_model(query, model, tokenizer, max_new_tokens=100):

    full_prompt = f"""
        You are an expert assistant. You know how to answer general knowledge questions. Your task is to provide a clear, accurate, and concise response to the question or instruction.
        You should only get one question and answer one time.
        Respond using the exact format shown below. Do not answer more than one question.
        ### Question: {query}
        ### Response:
    """

    inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = extract_answer(answer)
    return answer
