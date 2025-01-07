import pandas as pd
import os
import json

# Function to load annotations from Excel file (Train.xlsx, Test.xlsx, dev.xlsx)
def load_annotations_from_excel(excel_file):
    print(f"Loading annotations from: {excel_file}")
    data = pd.read_excel(excel_file)
    annotations = {}
    for idx, row in data.iterrows():
        video_name = row['name']
        gloss = row['gloss']
        translation = row['text']
        annotations[video_name] = {"gloss": gloss, "text": translation}
    print(f"Loaded {len(annotations)} annotations from {excel_file}")
    return annotations

# Function to build vocabulary from glosses and translations (in lowercase)
def build_vocab(annotations, include_translations=False):
    vocab = {}
    index = 1  # Start indexing from 1 (0 can be reserved for padding or unknown words)

    # Build vocabulary from glosses
    for annotation in annotations.values():
        gloss_words = annotation['gloss'].lower().split()  # Convert gloss to lowercase and split into words
        for word in gloss_words:
            if word not in vocab:
                vocab[word] = index
                index += 1

    # Include translations in the vocabulary
    if include_translations:
        for annotation in annotations.values():
            translation_words = annotation['text'].lower().split()  # Convert translation to lowercase and split into words
            for word in translation_words:
                if word not in vocab:
                    vocab[word] = index
                    index += 1

    return vocab

# Function to save vocabulary to a JSON file
def save_vocab(vocab, vocab_file):
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"Vocabulary saved to {vocab_file}")

# Main function to create and save vocabulary from Train.xlsx, Test.xlsx, and dev.xlsx
def create_vocab_from_dataset(train_file, test_file, dev_file, vocab_file, include_translations=True):
    # Load annotations from Train, Test, and Dev datasets
    train_annotations = load_annotations_from_excel(train_file)
    test_annotations = load_annotations_from_excel(test_file)
    dev_annotations = load_annotations_from_excel(dev_file)

    # Merge all annotations
    all_annotations = {**train_annotations, **test_annotations, **dev_annotations}

    # Build the vocabulary from the merged annotations
    vocab = build_vocab(all_annotations, include_translations=include_translations)

    # Save the vocabulary to a file
    save_vocab(vocab, vocab_file)

if __name__ == "__main__":
    # Paths to your dataset files (Train.xlsx, Test.xlsx, dev.xlsx)
    train_file = os.path.join('..', 'data', 'Train.xlsx')
    test_file = os.path.join('..', 'data', 'Test.xlsx')
    dev_file = os.path.join('..', 'data', 'dev.xlsx')
    
    # Path to save the vocabulary
    vocab_file = os.path.join('..', 'data', 'vocab.json')
    
    # Create vocabulary from dataset including both glosses and translations
    create_vocab_from_dataset(train_file, test_file, dev_file, vocab_file, include_translations=True)
