# build_vocab.py
import pandas as pd
import unicodedata
import re
import nltk
import os
from nltk.tokenize import WordPunctTokenizer
import json

# Verify NLTK version
print("NLTK version:", nltk.__version__)

# Ensure 'punkt' data is downloaded
nltk.download('punkt')

# Initialize tokenizer
tokenizer = WordPunctTokenizer()

def replace_umlauts(text):
    """
    Replaces German umlauts with their equivalent character sequences.
    """
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

def normalize_text(text):
    """
    Normalizes text by applying consistent preprocessing steps.
    """
    # Ensure text is string
    if not isinstance(text, str):
        text = str(text)
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    # Convert to lowercase
    text = text.lower()
    # Replace umlauts
    text = replace_umlauts(text)
    # Remove punctuation (except for periods)
    text = re.sub(r'[^\w\s\.]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """
    Tokenizes text using WordPunctTokenizer.
    """
    tokens = tokenizer.tokenize(text)
    return tokens

def main():
    # Paths to your Excel files
    train_excel = os.path.join('..', 'data', 'Train.xlsx')
    dev_excel = os.path.join('..', 'data', 'Test.xlsx')
    test_excel = os.path.join('..', 'data', 'dev.xlsx')

    # Load the data
    train_data = pd.read_excel(train_excel)
    dev_data = pd.read_excel(dev_excel)
    test_data = pd.read_excel(test_excel)

    # Concatenate the text and gloss data from all datasets
    all_texts = pd.concat([
        train_data['text'], train_data['gloss'],
        dev_data['text'], dev_data['gloss'],
        test_data['text'], test_data['gloss']
    ], ignore_index=True)

    # Drop NaN values
    all_texts = all_texts.dropna()

    # Apply normalization to all texts
    all_texts = all_texts.apply(normalize_text)

    # Tokenize all texts
    all_tokens = all_texts.apply(tokenize_text)

    # Build the vocabulary
    vocab_set = set()
    for tokens in all_tokens:
        vocab_set.update(tokens)

    print(f"Total unique words: {len(vocab_set)}")

    # Define special tokens (ensure consistency with your code)
    special_tokens = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3
    }

    # Assign IDs to words
    start_id = len(special_tokens)
    sorted_vocab = sorted(vocab_set)
    vocab = {word: idx + start_id for idx, word in enumerate(sorted_vocab)}
    vocab = {**special_tokens, **vocab}

    print(f"Total vocabulary size (including special tokens): {len(vocab)}")

    # Save the vocabulary
    vocab_file = os.path.join('..', 'data', 'vocab.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"Vocabulary saved to {vocab_file}")

if __name__ == "__main__":
    main()

