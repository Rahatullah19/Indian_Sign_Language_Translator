import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from wer import wer
import logging
import pandas as pd
from utils import extract_video_frames, extract_visual_token
from models.transformer import SignLanguageTransformer

# Setup logging
logging.basicConfig(
    filename=os.path.join('..', 'logs', 'evaluation.log'),
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def log_message(message):
    logging.info(message)
    print(message)

# Load annotations from Excel, including glosses and text
def load_annotations_from_excel(excel_file):
    log_message(f"Loading annotations from: {excel_file}")
    data = pd.read_excel(excel_file)
    annotations = {}
    for idx, row in data.iterrows():
        video_name = row['name']
        gloss = row['gloss']
        translation = row['text']
        annotations[video_name] = {"gloss": gloss, "text": translation}
    log_message(f"Loaded {len(annotations)} annotations.")
    return annotations

# Evaluate model performance on gloss and text prediction
def evaluate(model, excel_file, video_dir):
    log_message("************************************************************")
    log_message(f"Starting evaluation on dataset: {excel_file}")

    annotations = load_annotations_from_excel(excel_file)
    
    gloss_bleu_scores = []
    text_bleu_scores = []
    rouge_scores = []
    wer_scores = []

    for video_name, annotation in annotations.items():
        video_path = os.path.join(video_dir, video_name + '.mp4')  # Assuming the videos are in .mp4 format
        log_message(f"Processing video: {video_path}")
        
        if os.path.exists(video_path):
            frames = extract_video_frames(video_path)  # Extract frames from the video
            visual_tokens = extract_visual_token(frames)  # Generate visual tokens
            
            # Fetch the ground truth gloss and text
            ground_truth_gloss = annotation['gloss']
            ground_truth_text = annotation['text']
            
            # Predict glosses and text from the model
            predicted_gloss, predicted_text = model(visual_tokens, None, None, None, None)

            # Calculate BLEU score for glosses
            gloss_bleu_score = sentence_bleu([ground_truth_gloss.split()], predicted_gloss.split())
            gloss_bleu_scores.append(gloss_bleu_score)

            # Calculate BLEU score for text translation
            text_bleu_score = sentence_bleu([ground_truth_text.split()], predicted_text.split())
            text_bleu_scores.append(text_bleu_score)

            # Calculate ROUGE score for text translation
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_score = scorer.score(ground_truth_text, predicted_text)
            rouge_scores.append(rouge_score['rougeL'].fmeasure)

            # Calculate WER (Word Error Rate) for text translation
            wer_score = wer(ground_truth_text, predicted_text)
            wer_scores.append(wer_score)

            log_message(f"Video processed. Gloss BLEU: {gloss_bleu_score:.2f}, Text BLEU: {text_bleu_score:.2f}, ROUGE: {rouge_score['rougeL'].fmeasure:.2f}, WER: {wer_score:.2f}")
        
        else:
            log_message(f"Video file {video_path} not found.")

    # Log the average scores
    avg_gloss_bleu = sum(gloss_bleu_scores) / len(gloss_bleu_scores) if gloss_bleu_scores else 0
    avg_text_bleu = sum(text_bleu_scores) / len(text_bleu_scores) if text_bleu_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0

    log_message("************************************************************")
    log_message(f"Evaluation completed.")
    log_message(f"Average Gloss BLEU-4: {avg_gloss_bleu:.2f}")
    log_message(f"Average Text BLEU-4: {avg_text_bleu:.2f}")
    log_message(f"Average ROUGE: {avg_rouge:.2f}")
    log_message(f"Average WER: {avg_wer:.2f}")
    log_message("************************************************************")

# Entry point for running from the command line
if __name__ == "__main__":
    test_excel = os.path.join('..', 'data', 'Test.xlsx')      # Path to Test.xlsx
    video_dir = os.path.join('..', 'data', '')            # Path to test video files

    # Load your trained model here
    model = SignLanguageTransformer(visual_dim=512, emotion_dim=512, gesture_dim=512, gloss_vocab_size=1000, text_vocab_size=10000)

    evaluate(model, test_excel, video_dir)
