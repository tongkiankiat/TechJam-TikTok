# TechJam-TikTok

## Project Overview

NoiseGuard is an ML-powered system designed to make location-based reviews more trustworthy by filtering out noise; spam, irrelevant, and low-quality content, before it reaches users.

Online reviews directly influence how people choose restaurants, shops, and services. However, fake promotions, irrelevant comments, and copy-paste spam distort trust and harm both users and businesses. NoiseGuard solves this by combining machine learning with rule-based heuristics to automatically enforce content policies and highlight only authentic, relevant reviews.

With NoiseGuard, users get clearer insights, businesses enjoy fairer representation, and platforms reduce the manual burden of moderation while strengthening credibility.

## Training Results
### Model Training Results
![alt_text](https://github.com/tongkiankiat/TechJam-TikTok/blob/main/images/model-trained-results.jpeg)

### TF-IDF Trained Results
![alt_text](https://github.com/tongkiankiat/TechJam-TikTok/blob/main/images/tf-idf-results.png)

## Setup
- Install all dependencies
```bash
 pip install -r requirements.txt
```

## Running an Inference
- Locate inference.ipynb  
```bash
cd model
```
- Edit the sample review and run the cell to generate an inference
```python
# Model: mDeBERTa-v3-base
# Load model directly
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, re, joblib
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix, hstack

# TODO: Add the review to test here
sample = {
  "company_name": "McDonald's",
  "review_date": "2025-08-29",
  "text": "The new Spicy Tomato McChicken Set is wonderful for my wallet. The potato pops also go really well with it.",
  "stars": 5,
  "category": "food and beverages"
}
```
