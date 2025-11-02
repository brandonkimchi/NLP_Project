import re
import unicodedata
import html
import emoji
import pandas as pd
from pathlib import Path
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK assets if not present
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load spaCy English model (lightweight)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

STOPWORDS = set(stopwords.words("english"))
LEM = WordNetLemmatizer()

SLANG_MAP = {
    "im": "i am", "ive": "i have", "idk": "i do not know", "lol": "laugh out loud",
    "lmao": "laughing my ass off", "wtf": "what the fuck", "omg": "oh my god",
    "btw": "by the way", "tbh": "to be honest", "u": "you", "ur": "your",
    "dont": "do not", "cant": "can not", "wont": "will not", "gonna": "going to",
    "wanna": "want to", "gotta": "got to", "aint": "is not", "idc": "i do not care"
}


def normalize_unicode(text: str) -> str:
    """Ensure consistent unicode representation and decode HTML entities."""
    text = unicodedata.normalize("NFC", str(text))
    text = html.unescape(text)
    return text


def clean_text(text: str) -> str:
    """Normalize, expand slang, demojize, tokenize, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""

    # Unicode + HTML normalization
    t = normalize_unicode(text)

    # Remove HTML tags like <br />, etc.
    t = re.sub(r"<[^>]+>", " ", t)

    # Replace emojis with text names
    t = emoji.demojize(t, delimiters=(" ", " "))

    # Remove control / invisible chars
    t = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", t)

    # Normalize spaces and punctuation
    t = re.sub(r"[^\w\s!?',.:;()\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()

    # Expand slang
    for k, v in SLANG_MAP.items():
        t = re.sub(rf"\b{k}\b", v, t)

    # Tokenize + Lemmatize + Stopword Removal
    doc = nlp(t)
    tokens = []
    for tok in doc:
        lemma = LEM.lemmatize(tok.lemma_)
        if lemma not in STOPWORDS and lemma.strip():
            tokens.append(lemma)

    return " ".join(tokens)


def run(inp="data/IMDB_Dataset.csv", out="data/IMDB_clean.csv"):
    print(f"Reading {inp} ...")
    df = pd.read_csv(inp)

    if "review" not in df.columns:
        raise KeyError("Expected a 'review' column in the input CSV.")

    # Apply cleaning
    print("Cleaning and preprocessing reviews...")
    df["text"] = df["review"].astype(str).map(clean_text)

    # Map sentiment labels
    if "sentiment" in df.columns:
        df["label"] = (df["sentiment"].astype(str).str.lower() == "positive").astype(int)

    # Drop empties / duplicates
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["text"])

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(out, index=False)
    print(f"Saved cleaned dataset -> {out} ({len(df)} rows)")

    # Display one example
    print("\nSample cleaned review:\n", df.text.iloc[0][:300], "...")


if __name__ == "__main__":
    run()