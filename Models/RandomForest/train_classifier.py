# train_classifier.py

from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_data(pickle_path):
    """
    Load data from a pickle file.

    Args:
    - pickle_path (Path): Path to the pickle file.

    Returns:
    - data_dict (dict): Dictionary containing 'data' and 'labels'.
    """
    try:
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"Successfully loaded data from {pickle_path}.")
        return data_dict
    except FileNotFoundError:
        print(f"Error: The file {pickle_path} does not exist.")
        exit(1)
    except pickle.UnpicklingError:
        print(f"Error: The file {pickle_path} could not be unpickled.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the pickle file: {e}")
        exit(1)

def map_labels(labels):
    """
    Convert string labels to numeric labels using LabelEncoder.

    Args:
    - labels (list of str): List of label strings.

    Returns:
    - numeric_labels (np.ndarray): Array of numeric labels.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
    """
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    print("Labels have been converted to numeric class labels.")
    print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    return numeric_labels, label_encoder

def preprocess_data(data, labels):
    """
    Convert data and labels to NumPy arrays and ensure consistency.

    Args:
    - data (list of list of float): Feature vectors.
    - labels (list of str): Label strings.

    Returns:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Numeric label array.
    """
    # Determine the maximum length of feature vectors
    max_length = max(len(item) for item in data)
    print(f"Maximum feature vector length: {max_length}")

    # Pad shorter feature vectors with zeros to ensure uniform length
    data_padded = [item + [0.0]*(max_length - len(item)) if len(item) < max_length else item[:max_length] for item in data]
    
    X = np.array(data_padded)
    y = np.array(labels)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label array shape: {y.shape}")

    # Validate consistency
    if X.shape[0] != y.shape[0]:
        print("Error: The number of samples in data and labels do not match.")
        exit(1)
    print("Data and labels are consistent.")
    return X, y

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.

    Args:
    - X_train (np.ndarray): Training feature matrix.
    - y_train (np.ndarray): Training labels.

    Returns:
    - model (RandomForestClassifier): Trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest model has been trained.")
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model on the test set.

    Args:
    - model (RandomForestClassifier): Trained model.
    - X_test (np.ndarray): Test feature matrix.
    - y_test (np.ndarray): Test labels.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.

    Returns:
    - metrics (dict): Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Model Precision: {precision * 100:.2f}%")
    print(f"Model Recall: {recall * 100:.2f}%")
    print("\nClassification Report:\n", class_report)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    return metrics

def save_model(model, label_encoder, save_path):
    """
    Save the trained model and label encoder to a pickle file.

    Args:
    - model (RandomForestClassifier): Trained model.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
    - save_path (Path): Path to save the pickle file.
    """
    try:
        with open(save_path, 'wb') as f:
            pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
        print(f"Model and label encoder have been saved to {save_path}.")
    except Exception as e:
        print(f"Error saving the model: {e}")
        exit(1)

def save_metrics(metrics, save_path):
    """
    Save evaluation metrics to a JSON file.

    Args:
    - metrics (dict): Dictionary containing evaluation metrics.
    - save_path (Path): Path to save the JSON file.
    """
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = metrics.copy()
    if isinstance(metrics_serializable.get('confusion_matrix'), np.ndarray):
        metrics_serializable['confusion_matrix'] = metrics_serializable['confusion_matrix'].tolist()
    
    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    print(f"Metrics have been saved to {save_path}.")

def plot_confusion_matrix(conf_matrix, label_encoder, save_path=None):
    """
    Plot the confusion matrix.

    Args:
    - conf_matrix (np.ndarray): Confusion matrix.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
    - save_path (Path, optional): Path to save the plot.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}.")
    plt.show()

def plot_metrics(metrics, save_path=None):
    """
    Plot evaluation metrics: Accuracy, Precision, Recall.

    Args:
    - metrics (dict): Dictionary containing evaluation metrics.
    - save_path (Path, optional): Path to save the plot.
    """
    labels = ['Accuracy', 'Precision', 'Recall']
    scores = [metrics['accuracy'], metrics['precision'], metrics['recall']]

    plt.figure(figsize=(8,6))
    sns.barplot(x=labels, y=scores, palette='viridis')
    plt.ylim(0, 1)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    for i, score in enumerate(scores):
        plt.text(i, score + 0.01, f"{score*100:.2f}%", ha='center')
    if save_path:
        plt.savefig(save_path)
        print(f"Evaluation metrics plot saved to {save_path}.")
    plt.show()

def main():
    # Define paths
    pickle_file_path = Path('../Trained_Results/Models/Chinese_digit_data.pickle')
    model_save_path = Path('../Trained_Results/Models/Chinese_digit_model_sign.pickle')
    metrics_json_path = Path('../Trained_Results/Models/Chinese_digit_metrics.json')
    metrics_plot_path = Path('../Visualize_Output/Chinese_digit_metrics_plot.png')
    confusion_matrix_plot_path = Path('../Visualize_Output/Chinese_digit_confusion_matrix_plot.png')

    # Load data
    data_dict = load_data(pickle_file_path)

    # Inspect data_dict (optional but recommended for debugging)
    print("\n=== Inspecting data_dict ===")
    print("Keys in data_dict:", data_dict.keys())

    for key in data_dict.keys():
        print(f"\nKey: '{key}'")
        print(f"Type: {type(data_dict[key])}")
        try:
            print(f"Length: {len(data_dict[key])}")
            if isinstance(data_dict[key], list) and len(data_dict[key]) > 0:
                print(f"First 3 items: {data_dict[key][:3]}")
        except TypeError:
            print("Cannot determine length.")

    # Check if 'data' and 'labels' exist and are non-empty
    if 'data' not in data_dict:
        print("Error: 'data' key not found in data_dict.")
        exit(1)
    if 'labels' not in data_dict:
        print("Error: 'labels' key not found in data_dict.")
        exit(1)
    if not data_dict['labels']:
        print("Error: 'labels' list is empty.")
        exit(1)

    data = data_dict['data']
    labels = data_dict['labels']

    # Preprocess data
    X, y = preprocess_data(data, labels)

    # Convert labels to numeric labels
    numeric_labels, label_encoder = map_labels(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Train Random Forest model
    model = train_model(X_train, y_train)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, label_encoder)

    # Save the model and label encoder
    save_model(model, label_encoder, model_save_path)

    # Save evaluation metrics to JSON
    save_metrics(metrics, metrics_json_path)

    # Plot evaluation metrics
    plot_metrics(metrics, save_path=metrics_plot_path)

    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], label_encoder, save_path=confusion_matrix_plot_path)

if __name__ == "__main__":
    main()
