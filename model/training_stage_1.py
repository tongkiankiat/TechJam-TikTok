# Training Script
import os, math, torch, evaluate, csv, json, time
import numpy as np

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict

# Constants
MODEL_NAME = "microsoft/mdeberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NO_OF_LABELS = 3
CATEGORIES = ["Food and Beverages", "Education", "Healthcare", "Retail", "Arts", "Hotels"]
DATA_FILE_PATH = "/content/sample_data/withtfidf.jsonl"
OUTPUT_DIR = "/content/sample_data/loc-review-classification-model-v2"
MAX_LENGTH = 256
SEED = 42

# Label Mappings
id2label = {0: "spam", 1: "irrelevant", 2: "relevant"}
label2id = {value: key for key, value in id2label.items()}

start = time.time()
# Load model to train
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
classification_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NO_OF_LABELS,
    id2label=id2label,
    label2id=label2id
)

# Load dataset
DATA_FILES = {
    "trainval": DATA_FILE_PATH
}
raw = load_dataset("json", data_files=DATA_FILES)["trainval"]

required_headers = {"company_name", "review_date", "text", "stars", "category", "label"}
missing = required_headers - set(raw.features.keys())
if missing:
  ValueError(f"Dataset missing required keys: {missing}")

# Special tokens
special_tokens = []
categories = set()
for ex in raw:
  categories.add(ex["category"])
# Add all categories
special_tokens += [f"[CAT_{c}]" for c in sorted(categories)]

# Add all ratings
special_tokens += [f"[RATING_{r}]" for r in range(1, 6)]

tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# Preprocessing
def construct_pair(datarow):
  cat = datarow["category"]
  rating = datarow["stars"]

  company = datarow["company_name"].strip()
  poi = f"POI: {company} [CAT_{cat}] [RATING_{rating}]"

  text = datarow["text"]
  if text is None:
    text = ""
  else:
    text = str(text).strip()

  encoded = tokenizer(
      text,
      poi,
      truncation=True,
      max_length=MAX_LENGTH
  )

  y = datarow["label"]
  encoded["labels"] = int(y)
  return encoded

processed = raw.map(construct_pair, remove_columns=raw.column_names)

# Splitting dataset
processed = processed.class_encode_column("labels")
splits = processed.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="labels")
dataset = DatasetDict({
    "train": splits["train"],
    "validation": splits["test"]
})

# Evaluation
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=-1)
  return {
      "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
      "precision": precision.compute(predictions=preds, references=labels, average="macro")["precision"],
      "recall_macro": recall.compute(predictions=preds, references=labels, average="macro")["recall"],
      "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
  }

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    fp16=torch.cuda.is_available(),
    seed=SEED,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=classification_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
end = time.time()
elapsed_time = (end - start) / 3600
print(f"Time to train model: {elapsed_time:.2f} seconds")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete. Best model saved to:", OUTPUT_DIR)