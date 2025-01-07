import os
import pandas as pd
import cv2

# Load annotations from Excel, including glosses and text
def load_annotations_from_excel(excel_file):
    print(f"Loading annotations from: {excel_file}")
    data = pd.read_excel(excel_file)
    annotations = {}
    for idx, row in data.iterrows():
        video_name = row['name']
        gloss = row['gloss']
        translation = row['text']
        annotations[video_name] = {"gloss": gloss, "text": translation}
    print(f"Loaded {len(annotations)} annotations.")
    return annotations

# Extract frames from videos
def extract_frames(video_path, output_dir):
    print(f"Extracting frames from video: {video_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        if count % 10 == 0:  # Print every 10 frames
            print(f"Extracted {count} frames...")
    
    cap.release()
    print(f"Finished extracting {count} frames from {video_path}")

# Preprocess the dataset by extracting frames and loading annotations
def preprocess_data(excel_file, video_dir, output_dir):
    print("Starting preprocessing...")
    annotations = load_annotations_from_excel(excel_file)
    
    for video_name in annotations.keys():
        video_path = os.path.join(video_dir, video_name + '.mp4')  # Assuming videos are in .mp4 format
        output_subdir = os.path.join(output_dir, os.path.splitext(video_name)[0])
        
        if os.path.exists(video_path):
            extract_frames(video_path, output_subdir)
        else:
            print(f"Video file {video_path} not found.")
    
    print(f"Preprocessing completed. Frames stored in {output_dir}")

# Entry point for running from the command line
if __name__ == "__main__":
    excel_file_tr = os.path.join('data', 'Train.xlsx')  
    excel_file_t = os.path.join('data', 'Test.xlsx')  
    excel_file_d = os.path.join('data', 'dev.xlsx')  
    video_dir = os.path.join('data', '')        # Use os.path.join for video directory
    output_dir = os.path.join('data', 'output_frames') # Use os.path.join for output directory

    preprocess_data(excel_file_tr, video_dir, output_dir)
    preprocess_data(excel_file_t, video_dir, output_dir)
    preprocess_data(excel_file_d, video_dir, output_dir)
