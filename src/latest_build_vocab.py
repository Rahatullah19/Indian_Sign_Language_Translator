# Example script to build vocab.json

import json
from utils_1 import load_annotations_from_excel, normalize_text

def build_vocab(excel_files, vocab_file):
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<OOV>': 3}
    for excel in excel_files:
        annotations = load_annotations_from_excel(excel)
        for annotation in annotations.values():
            for key in ['text', 'gloss']:
                text = annotation[key]
                text = ' '.join(text) if isinstance(text, list) else text
                text = normalize_text(text)
                for word in text.split():
                    if word not in vocab:
                        vocab[word] = len(vocab)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

build_vocab(['data/Train.xlsx', 'data/dev.xlsx', 'data/Test.xlsx'], 'data/vocabfull.json')