import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, RobertaTokenizer
from datasets import Dataset

# Load training data
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

# Load tokenizer and model
model_name = "Salesforce/codet5-small"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Format dataset for training
def format_data(df):
    inputs = ["generate test: " + code for code in df["function_code"].tolist()]
    targets = df["test_code"].tolist()
    return Dataset.from_dict({"input": inputs, "target": targets})

train_dataset = format_data(train_df)
val_dataset = format_data(val_df)

# Tokenize data
def tokenize_data(examples):
    model_inputs = tokenizer(examples["input"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["target"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize_data, batched=True)
val_dataset = val_dataset.map(tokenize_data, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model_output",
    evaluation_strategy="epoch",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=35,
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("Training complete! Model saved in './trained_model'")
