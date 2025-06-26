from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch
import pandas as pd
from datasets import Dataset, DatasetDict


label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

df_train = pd.read_csv("/content/drive/MyDrive/ATG_Assignment2/emotion_train.csv")
df_test = pd.read_csv("/content/drive/MyDrive/ATG_Assignment2/emotion_test.csv")
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})



tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Split into train/test
split_datasets = dataset["train"].train_test_split(test_size=0.2)


tokenized_datasets = split_datasets.map(tokenize_function, batched=True)


tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Load base model and apply LoRA
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6
)


# LoRA config (optional but efficient)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "v_lin"]
)

model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs"
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 7. Train and save
trainer.train()
trainer.save_model("/content/drive/MyDrive/ATG_Assignment2/finetuned_emotion_model")
tokenizer.save_pretrained("/content/drive/MyDrive/ATG_Assignment2/finetuned_emotion_model")
