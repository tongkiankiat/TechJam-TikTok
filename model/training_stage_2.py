# LogisticRegression Classifier with TF-IDF

import json
import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib

MODEL_NAME = "kiankiat/loc-review-classification-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE_PATH = "/content/sample_data/withtfidf.jsonl"
SEED = 42
MAX_LENGTH = 256
BATCH = 64
TF_IDF_COL = "tfidf_score"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
classification_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

DATA_FILES = {
    "trainval": DATA_FILE_PATH
}
raw = load_dataset("json", data_files=DATA_FILES)["trainval"]
raw = raw.class_encode_column("label")
splits = raw.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")
train_ds, val_ds = splits["train"], splits["test"]

def tfidf_row_from_json_string(s):
    arr = np.asarray(json.loads(s), dtype=np.float32).reshape(1, -1)
    return csr_matrix(arr)

def build_tfidf_matrix(dataset, col=TF_IDF_COL):
    rows = [tfidf_row_from_json_string(s) for s in dataset[col]]
    return vstack(rows, format="csr")

Xtr_tfidf = build_tfidf_matrix(train_ds)
Xva_tfidf = build_tfidf_matrix(val_ds)

def build_poi(company, category, stars):
    return f"POI: {str(company).strip()} [CAT_{category}] [RATING_{stars}]"

@torch.inference_mode()
def probs_from_transformer(ds):
    out = []
    n = len(ds)
    for i in range(0, n, BATCH):
        t = ds["text"][i:i+BATCH]
        p = [build_poi(c, g, s) for c, g, s in zip(ds["company_name"][i:i+BATCH],
                                                  ds["category"][i:i+BATCH],
                                                  ds["stars"][i:i+BATCH])]
        enc = tokenizer(t, p, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        logits = classification_model(**enc).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out)

p_train = probs_from_transformer(train_ds)
p_val   = probs_from_transformer(val_ds)

# Stack features: [TFIDF || probs]
Xtr_meta = hstack([Xtr_tfidf, p_train])
Xva_meta = hstack([Xva_tfidf, p_val])
y_tr = np.array(train_ds["label"])
y_va = np.array(val_ds["label"])

# Meta-classifier
meta = LogisticRegression(max_iter=1000, n_jobs=-1, solver="saga", multi_class="auto")
meta.fit(Xtr_meta, y_tr)

pred = meta.predict(Xva_meta)
print(classification_report(y_va, pred, digits=4))
print("Macro F1:", f1_score(y_va, pred, average="macro"))

joblib.dump(meta, "/content/sample_data/meta_classifier.joblib")
print("Done!")