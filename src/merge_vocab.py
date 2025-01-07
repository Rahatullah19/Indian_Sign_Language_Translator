import json
import os

def load_vocab(file_path):
    """
    Loads a vocabulary JSON file.
    
    Args:
        file_path (str): Path to the vocab JSON file.
        
    Returns:
        dict: Dictionary mapping tokens to their IDs.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping.")
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def merge_vocabs(vocab1, vocab2):
    """
    Merges two vocabularies, giving priority to vocab1 in case of duplicates.
    
    Args:
        vocab1 (dict): First vocabulary dictionary.
        vocab2 (dict): Second vocabulary dictionary.
        
    Returns:
        dict: Merged vocabulary dictionary.
    """
    merged_vocab = vocab1.copy()
    for token, idx in vocab2.items():
        if token not in merged_vocab:
            merged_vocab[token] = len(merged_vocab)
    return merged_vocab

def save_vocab(vocab, file_path):
    """
    Saves the vocabulary dictionary to a JSON file.
    
    Args:
        vocab (dict): Vocabulary dictionary.
        file_path (str): Path to save the merged vocab JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"Merged vocabulary saved to {file_path}")

def main():
    # Define file paths
    vocab_old_path = os.path.join('data', 'vocabfull.json')  # Adjust path if necessary
    vocab_new_path = os.path.join('data', 'vocab.json')      # Current vocab.json
    merged_vocab_path = os.path.join('data', 'vocab_merged.json')  # Output file
    
    # Load vocabularies
    print("Loading vocabularies...")
    vocab_old = load_vocab(vocab_old_path)
    vocab_new = load_vocab(vocab_new_path)
    
    # Merge vocabularies
    print("Merging vocabularies...")
    merged_vocab = merge_vocabs(vocab_new, vocab_old)
    
    # Save merged vocabulary
    print("Saving merged vocabulary...")
    save_vocab(merged_vocab, merged_vocab_path)
    
    # Optionally, replace the old vocab.json with the merged one
    replace = input("Do you want to replace the original vocab.json with the merged vocabulary? (y/n): ")
    if replace.lower() == 'y':
        os.remove(vocab_new_path)
        os.rename(merged_vocab_path, vocab_new_path)
        print("Original vocab.json has been replaced with the merged vocabulary.")
    else:
        print("Merged vocabulary saved separately. No changes made to the original vocab.json.")

if __name__ == "__main__":
    main()