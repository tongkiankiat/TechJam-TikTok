# Model: mDeBERTa-v3-base
# Load model directly
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Constants
MODEL_NAME = "microsoft/mdeberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NO_OF_LABELS = 3

# Define the label mappings (required for classification models)
id2label = {0: "spam", 1: "irrelevant", 2: "relevant"}
label2id = {value: key for key, value in id2label.items()}

# Load the models
tokenizer_model = AutoTokenizer.from_pretrained(MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NO_OF_LABELS,
    id2label=id2label,
    label2id=label2id
).to(DEVICE).eval()

def classify_review(review_text: str, max_length: int = 256):
  if not isinstance(review_text, str):
    raise TypeError("Please submit a string for the review")

  tokenized_inputs = tokenizer_model(
      review_text,
      return_tensors="pt", # This tells the model to return PyTorch tensors (pt). We can also return TensorFlow arrays (tf) or NumPy arrays (np)
      padding=True,
      truncation=True,
      max_length=max_length
  )
  tokenized_inputs.to(DEVICE) # Fix bug where some tensors were still on CPU

  with torch.inference_mode():
    logits = classifier_model(**tokenized_inputs).logits

  probabilities = torch.softmax(logits, dim=-1).squeeze(0).tolist() # Softmax function is usually used for classification, as it returns values from 0 to 1, which we treat as probabilities. Squeeze is used to remove any dimensions that are not utilised
  prediction_id = int(torch.argmax(logits, dim=-1).item()) # This will return an array of with 3 indices (since we have 3 labels). The index with the highest value will be the label that we predict. torch.argmax picks the highest value for us
  
  # return id2label[prediction_id]
  return {"label": id2label[prediction_id], "probs": {id2label[i]: float(p) for i, p in enumerate(probabilities)}}

if __name__ == "__main__":
  review = str(input("Enter review: ")).strip()
  result = classify_review(review)
  print(result)