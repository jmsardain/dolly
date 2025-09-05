import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from data_preparation import loaddata, format_dolly
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from rag import getRAG, retrieve
from transformers import pipeline
from evaluate_methods import evaluate_model

def main(train, evaluate):

    model_name = "mistralai/Mistral-7B-v0.1"
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token ## important to set here

    # print(" ======== Get pretrained model ========")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )

    # print(" ======== Get dataset ========")
    train_dataset, val_dataset, all_instructions = loaddata(name="databricks/databricks-dolly-15k")
    tokenized_train = train_dataset.map(lambda ex: format_dolly(ex, tokenizer))
    tokenized_val   = val_dataset.map(lambda ex: format_dolly(ex, tokenizer))

    if train:
        base_model = prepare_model_for_kbit_training(base_model)
        ## Use LoRA for small fine tuning
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, peft_config)

        training_args = TrainingArguments(
            output_dir="./mistral_dolly_finetune",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=50,
            save_steps=200,
            eval_steps=200,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

        trainer.train()
        model.save_pretrained("./mistral_dolly_lora")


    if evaluate:
        ## Add RAG to test model without it as well
        model_RAG, index = getRAG('all-MiniLM-L6-v2', all_instructions) ## do RAG on whole datasets
        model = PeftModel.from_pretrained(base_model, "./mistral_dolly_lora")

        # print("1) Base model, no RAG")
        # bleu1, f11, bertf11 = evaluate_model(base_model, val_dataset, index, tokenizer, model_RAG, all_instructions, use_rag=False, n=200)
        # with open("performance.txt", "a") as f:
        #     f.write("1) Base model, no RAG\n")
        #     f.write(f"BLEU {bleu1} F1: {f11}, BERT: {bertf11}\n")

        # print("2) Base model, RAG")
        # bleu2, f12, bertf12 = evaluate_model(base_model, val_dataset, index, tokenizer, model_RAG, all_instructions, use_rag=True, n=200)
        # with open("performance.txt", "a") as f:
        #     f.write("2) Base model, RAG\n")
        #     f.write(f"BLEU {bleu2} F1: {f12}, BERT: {bertf12}\n")

        print("3) Fine-tuned model, no RAG")
        bleu3, f13, bertf13 = evaluate_model(model, val_dataset, index, tokenizer, model_RAG, all_instructions, use_rag=False, n=200)
        with open("performance.txt", "a") as f:
            f.write("3) Fine-tuned model, no RAG\n")
            f.write(f"BLEU {bleu3} F1: {f13}, BERT: {bertf13}\n")

        # print("4) Fine-tuned model, RAG")
        # bleu4, f14, bertf14 = evaluate_model(model, val_dataset, index, tokenizer,  model_RAG, all_instructions, use_rag=True, n=200)
        # with open("performance.txt", "a") as f:
        #     f.write("4) Fine-tuned model, RAG\n")
        #     f.write(f"BLEU {bleu4} F1: {f14}, BERT: {bertf14}\n")


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', default=False, help="Fine tune the model")
    parser.add_argument("--evaluate", action='store_true', default=False, help="Evaluation")
    args = parser.parse_args()
    main(args.train, args.evaluate)
    pass
