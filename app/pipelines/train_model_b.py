from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
from app.config import settings
import os

# Dummy dataset
texts = ["I love this!", "I hate this!", "This is amazing", "This is terrible"]
labels = [1, 0, 1, 0]
dataset = Dataset.from_dict({"text": texts, "label": labels})

# Tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Training
training_args = TrainingArguments(
    output_dir="./app/models/model_b",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

trainer.train()

# Save model
os.makedirs(settings.model_b_path, exist_ok=True)
model.save_pretrained(settings.model_b_path)
tokenizer.save_pretrained(settings.model_b_path)
print(f"Model B saved to {settings.model_b_path}")
