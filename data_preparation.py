from datasets import load_dataset


def loaddata(name="databricks/databricks-dolly-15k"):
    dataset = load_dataset("databricks/databricks-dolly-15k")
    train_data = dataset["train"].train_test_split(test_size=0.1)
    return train_data["train"], train_data["test"], dataset["train"][:]["instruction"]

def format_dolly(example, tokenizer):
    instruction = example["instruction"]
    context = example.get("context", "")
    response = example["response"]
    if context:
        prompt = f"Instruction: {instruction}\nContext: {context}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"
    return {"input_ids": tokenizer(prompt, truncation=True, padding="max_length", max_length=512).input_ids,
            "labels": tokenizer(response, truncation=True, padding="max_length", max_length=512).input_ids}
