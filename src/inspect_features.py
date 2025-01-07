# inspect_features.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_feature(feature_path):
    """
    Loads a .npy feature file and returns the feature array.
    """
    if not os.path.exists(feature_path):
        print(f"Feature file not found: {feature_path}")
        return None

    features = np.load(feature_path)
    print(f"Loaded features from {feature_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature data type: {features.dtype}")
    return features

def inspect_feature_statistics(features, feature_name):
    """
    Prints basic statistics of the feature array.
    """
    print(f"\n{feature_name} Feature Statistics:")
    print(f"Mean: {np.mean(features):.6f}")
    print(f"Standard Deviation: {np.std(features):.6f}")
    print(f"Minimum Value: {np.min(features):.6f}")
    print(f"Maximum Value: {np.max(features):.6f}")

def plot_feature_over_time(features, feature_name):
    """
    Plots a selected dimension of the feature over time (frames).
    """
    num_dimensions = features.shape[1]
    feature_dimension = int(input(f"Select a feature dimension to plot (0 to {num_dimensions - 1}): "))
    if feature_dimension < 0 or feature_dimension >= num_dimensions:
        print("Invalid feature dimension selected.")
        return

    feature_values = features[:, feature_dimension]
    plt.figure(figsize=(10, 4))
    plt.plot(feature_values)
    plt.title(f'{feature_name} Feature Dimension {feature_dimension} Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Feature Value')
    plt.grid(True)
    plt.show()

def visualize_features_pca(features, feature_name):
    """
    Reduces the feature dimensions using PCA and visualizes them.
    """
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
    plt.title(f'PCA of {feature_name} Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

def check_for_nan_inf(features, feature_name):
    """
    Checks if the feature array contains NaN or infinite values.
    """
    if np.isnan(features).any():
        print(f"{feature_name} features contain NaN values!")
    else:
        print(f"{feature_name} features do not contain NaN values.")

    if np.isinf(features).any():
        print(f"{feature_name} features contain infinite values!")
    else:
        print(f"{feature_name} features do not contain infinite values.")

def normalize_features(features):
    """
    Normalizes the feature array using standard scaling.
    """
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

def main():
    # Set the paths to your feature files
    base_dir = 'data/features'
    video_name = input("Enter the sanitized video name (e.g., 'train_video1'): ")

    # Paths to the feature files
    visual_feature_path = os.path.join(base_dir, 'visual', f'{video_name}.npy')
    emotion_feature_path = os.path.join(base_dir, 'emotion', f'{video_name}.npy')
    gesture_feature_path = os.path.join(base_dir, 'gesture', f'{video_name}.npy')

    # Load features
    visual_features = load_feature(visual_feature_path)
    emotion_features = load_feature(emotion_feature_path)
    gesture_features = load_feature(gesture_feature_path)

    # Check if features are loaded
    if visual_features is None or emotion_features is None or gesture_features is None:
        print("One or more feature files could not be loaded. Please check the file paths.")
        return

    # Inspect features
    inspect_feature_statistics(visual_features, 'Visual')
    inspect_feature_statistics(emotion_features, 'Emotion')
    inspect_feature_statistics(gesture_features, 'Gesture')

    # Check for NaN or infinite values
    check_for_nan_inf(visual_features, 'Visual')
    check_for_nan_inf(emotion_features, 'Emotion')
    check_for_nan_inf(gesture_features, 'Gesture')

    # Normalize features (optional)
    normalize = input("\nDo you want to normalize the features? (yes/no): ").strip().lower()
    if normalize == 'yes':
        visual_features = normalize_features(visual_features)
        emotion_features = normalize_features(emotion_features)
        gesture_features = normalize_features(gesture_features)
        print("Features have been normalized using standard scaling.")

    # Plot features over time
    plot_choice = input("\nDo you want to plot a feature dimension over time? (yes/no): ").strip().lower()
    if plot_choice == 'yes':
        print("\nSelect feature type to plot:")
        print("1. Visual")
        print("2. Emotion")
        print("3. Gesture")
        feature_type = input("Enter your choice (1/2/3): ").strip()
        if feature_type == '1':
            plot_feature_over_time(visual_features, 'Visual')
        elif feature_type == '2':
            plot_feature_over_time(emotion_features, 'Emotion')
        elif feature_type == '3':
            plot_feature_over_time(gesture_features, 'Gesture')
        else:
            print("Invalid choice.")

    # Visualize features using PCA
    pca_choice = input("\nDo you want to visualize features using PCA? (yes/no): ").strip().lower()
    if pca_choice == 'yes':
        print("\nSelect feature type to visualize:")
        print("1. Visual")
        print("2. Emotion")
        print("3. Gesture")
        feature_type = input("Enter your choice (1/2/3): ").strip()
        if feature_type == '1':
            visualize_features_pca(visual_features, 'Visual')
        elif feature_type == '2':
            visualize_features_pca(emotion_features, 'Emotion')
        elif feature_type == '3':
            visualize_features_pca(gesture_features, 'Gesture')
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
