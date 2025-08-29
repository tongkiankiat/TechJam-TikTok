import pandas as pd
import re

# === CONFIG ===
MIN_WORDS = 5          # reviews with fewer than 5 words are spam
MIN_CHARS = 20         # reviews with fewer than 20 characters are spam

# Regex patterns for spammy content
SPAM_PATTERNS = [
    r"(https?:\/\/[^\s]+)",       # URLs
    r"(www\.[^\s]+)",             # www links
    r"(\b\d{2,}%\b)",             # discount codes like "50%"
]

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV into dataframe"""
    return pd.read_csv(file_path)

def check_regex_spam(text: str) -> bool:
    """Check if review text contains spammy patterns"""
    if pd.isna(text):  # handle NaN
        return True
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def check_length_spam(text: str) -> bool:
    """Check if review text is too short"""
    if pd.isna(text):
        return True
    words = text.split()
    return len(words) < MIN_WORDS or len(text) < MIN_CHARS

def detect_duplicates(df: pd.DataFrame) -> pd.Series:
    """
    Mark reviews as spam if duplicate text exists for the same business.
    Different authors but same text+business = spam.
    """
    return df.duplicated(subset=["business_name", "text"], keep=False)

def mark_spam(df: pd.DataFrame) -> pd.DataFrame:
    """Apply spam filters"""
    df["spam_regex"] = df["text"].apply(check_regex_spam)
    df["spam_length"] = df["text"].apply(check_length_spam)
    df["spam_duplicate"] = detect_duplicates(df)

    # Final spam label (any condition triggers spam)
    df["is_spam"] = df[["spam_regex", "spam_length", "spam_duplicate"]].any(axis=1)

    return df

if __name__ == "__main__":
    file_path = r"C:\Users\rayso\Downloads\googlekagglereviews\reviews.csv"
    df = load_data(file_path)
    df = mark_spam(df)

    # Save output
    df.to_csv(r"C:\Users\rayso\OneDrive\Documents\GitHub\TechJam-TikTok\datasets\reviews_with_spam_flags.csv", index=False)
    print("✅ Spam filtering complete. Output saved to reviews_with_spam_flags.csv")
