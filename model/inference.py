# Model: mDeBERTa-v3-base
# Load model directly
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, re, joblib
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix, hstack

# Constants
MODEL_NAME = "kiankiat/loc-review-classification-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the models
tokenizer_model = AutoTokenizer.from_pretrained(MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load the meta_classifier
meta_path = hf_hub_download(repo_id=MODEL_NAME, filename="meta_classifier.joblib")
meta = joblib.load(meta_path)

sample = {
  "company_name": "McDonald's",
  "review_date": "2025-08-29",
  "text": "The new Spicy Tomato McChicken Set is wonderful for my wallet. The potato pops also go really well with it.",
  "stars": 5,
  "category": "food and beverages"
}

vocab = [
    "the","i","of","was","to","a","for","in","is","it","that","at","you","my","on","with","but","this","about","its",
    "and","we","me","they","are","out","their","an","our","not","been","if","service","like","also","had","so","as",
    "your","all","have","ive","from","even","here","very","just","food","never","place","were","there","amazing",
    "honestly","experience","be","good","by","get","how","people","while","staff","new","say","heard","time","friend",
    "call","which","check","up","dont","or","more","code","can","great","deals","absolutely","youre","has","meanwhile",
    "when","discount","one","told","these","really","recently","exclusive","some","visit","crypto","what","im","no",
    "only","us","them","offer","any","best","now","would","recommend","singapore","care","weather","clinic","unbeatable",
    "got","where","will","help","loved","life","too","offers","looking","did","discovered","miss","day","off","cash",
    "well","made","highly","local","nothing","spent","love","www","ever","friendly","she","than",
    "services","back","quick","over","restaurant","nice","definitely","go","always","other","bar","last"
]

def remove_punct(text):
    text = text or ""
    return re.sub(r"[\'\"’.,:&@!#\-\(\)0-9–—-−]", "", text)

def remove_escape_chars(text: str) -> str:
    if text is None:
        return ""
    cleaned = re.sub(r'[\n\t\r\f\v]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def lowercase(text):
  return text.lower()

def text_to_array(review: str):
    if review is None:
        return [0.0] * len(vocab)
    # Normalize: lowercase, remove punctuation, split on whitespace
    review = remove_escape_chars(lowercase(remove_punct(review)))
    tokens = set(review.split())
    # Build the array
    return [1.0 if word in tokens else 0.0 for word in vocab]

sample["tfidf_score"] = text_to_array(sample["text"])

def tokenize_for_inference(datarow):
    cat = datarow["category"]
    rating = datarow["stars"]

    company = datarow["company_name"].strip()
    poi = f"POI: {company} [CAT_{cat}] [RATING_{rating}]"

    text = datarow["text"]
    if text is None:
        text = ""
    else:
        text = str(text).strip()

    encoded = tokenizer_model(
        text,
        poi,
        truncation=True,
        max_length=256,
        return_tensors="pt"   # so we can pass directly to model
    )
    return encoded

inputs = tokenize_for_inference(sample)

def tfidf_row(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    return csr_matrix(arr)

with torch.inference_mode():
  logits = classifier_model(**inputs).logits
  probs = torch.softmax(logits, dim=-1)
  X_tfidf = tfidf_row(sample["tfidf_score"])
  X_meta = hstack([X_tfidf, csr_matrix(probs)], format="csr")
  pred = meta.predict(X_meta)[0]

print("Transformer probs:", probs.tolist())
print("Meta prediction:", pred)