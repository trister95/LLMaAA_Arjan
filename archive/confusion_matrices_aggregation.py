import argparse
import json
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from itertools import product
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class NERPredictor:
    def __init__(self, model_name):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        filtered_tokens = [token for token in tokens if token not in self.tokenizer.all_special_tokens]
        filtered_predictions = [pred for token, pred in zip(tokens, predictions) if token not in self.tokenizer.all_special_tokens]
        
        return [self._get_most_common(labels) for _, labels in self._aggregate_tokens(filtered_tokens, filtered_predictions)]

    def _aggregate_tokens(self, tokens, predictions):
        word_predictions = defaultdict(list)
        current_word = ""
        for token, pred in zip(tokens, predictions):
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    yield current_word, word_predictions[current_word]
                    word_predictions[current_word] = []
                current_word = token
            word_predictions[current_word].append(self.id2label[pred.item()])
        if current_word:
            yield current_word, word_predictions[current_word]

    def _get_most_common(self, labels):
        return max(set(labels), key=labels.count)

def load_jsonl(file_path, key='tags'):
    results = []
    with open(file_path) as f:
        for i, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
                if key in data:
                    results.append(data[key])
                else:
                    print(f"Key '{key}' not found in line {i}: {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError in line {i}: {e.msg} (line: {line.strip()})")
            except Exception as e:
                print(f"Error processing line {i}: {e} (line: {line.strip()})")
    return results

def load_multiple_files(folder_path):
    file_pattern = os.path.join(folder_path, '*.jsonl')
    return [load_jsonl(file) for file in glob(file_pattern)]

def flatten_predictions(predictions):
    return [item for sublist in predictions for item in sublist]

def create_confusion_matrix(true_labels, pred_labels):
    all_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + epsilon)
    
    return cm, cm_normalized, all_labels

def plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, output_folder, use_absolute_numbers=False):
    plt.figure(figsize=(12, 10))
    
    if use_absolute_numbers:
        # Round the absolute numbers to integers
        cm_int = np.round(cm).astype(int)
        sns.heatmap(cm_normalized, annot=cm_int, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title(f'Confusion Matrix: {method1} vs {method2}')
    plt.xlabel(f'Predicted ({method2})')
    plt.ylabel(f'True ({method1})')
    plt.tight_layout()
    
    number_type = 'absolute' if use_absolute_numbers else 'relative'
    output_file = os.path.join(output_folder, f'confusion_matrix_{method1}_vs_{method2}_{number_type}.png')
    plt.savefig(output_file)
    plt.close()

def print_prediction_info(method, predictions):
    flat_preds = flatten_predictions(predictions)
    unique_labels = set(flat_preds)
    label_counts = {label: flat_preds.count(label) for label in unique_labels}
    print(f"{method} predictions:")
    print(f"  Total predictions: {len(flat_preds)}")
    print(f"  Unique labels: {len(unique_labels)}")
    print("  Label counts:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {label}: {count}")
    print()

def main(args):
    # Load human annotations
    D_hum_predictions = load_jsonl(args.d_human_annotations, 'tags')
    T_hum_predictions = load_jsonl(args.t_human_annotations, 'tags')
    
    # Load texts for LLLMaAA predictions
    texts = load_jsonl(args.d_human_annotations, 'text')
    
    # Generate LLLMaAA predictions
    D_predictor = NERPredictor(args.d_model)
    T_predictor = NERPredictor(args.t_model)
    D_lllmaaa_predictions = [D_predictor.predict(text) for text in texts]
    T_lllmaaa_predictions = [T_predictor.predict(text) for text in texts]

    # Load other predictions
    data = {
        "D_LLLMaAA": [D_lllmaaa_predictions],
        "T_LLLMaAA": [T_lllmaaa_predictions],
        "D_hum": [D_hum_predictions],
        "T_hum": [T_hum_predictions],
        "direct_demo1": load_multiple_files(args.direct_demo1),
        "direct_demo2": load_multiple_files(args.direct_demo2)
    }
    method_names = list(data.keys())

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    for method, predictions in data.items():
        for i, pred_set in enumerate(predictions):
            print(f"{method} - Set {i+1}")
            print_prediction_info(method, pred_set)

    # Generate confusion matrices
    for method1, method2 in product(method_names, repeat=2):
        if method1 != method2:
            print(f"Generating confusion matrix for {method1} vs {method2}")
            
            all_cm = []
            all_cm_normalized = []
            all_labels = set()
            
            # Create confusion matrices for all pairs of prediction files
            for preds1, preds2 in product(data[method1], data[method2]):
                flat_preds1 = flatten_predictions(preds1)
                flat_preds2 = flatten_predictions(preds2)
                
                try:
                    cm, cm_normalized, labels = create_confusion_matrix(flat_preds1, flat_preds2)
                    all_cm.append(cm)
                    all_cm_normalized.append(cm_normalized)
                    all_labels.update(labels)
                except Exception as e:
                    print(f"Error creating confusion matrix: {e}")
                    print(f"Skipping this pair and continuing...")
                    continue
            
            if not all_cm:
                print(f"No valid confusion matrices generated for {method1} vs {method2}. Skipping...")
                continue
            
            # Average the confusion matrices
            all_labels = sorted(all_labels)
            avg_cm = np.mean([np.pad(cm, ((0, len(all_labels) - cm.shape[0]), (0, len(all_labels) - cm.shape[1])), 
                                     mode='constant') for cm in all_cm], axis=0)
            avg_cm_normalized = np.mean([np.pad(cm, ((0, len(all_labels) - cm.shape[0]), (0, len(all_labels) - cm.shape[1])), 
                                     mode='constant') for cm in all_cm_normalized], axis=0)
            
            if args.output_type == 'both':
                plot_confusion_matrix(avg_cm, avg_cm_normalized, all_labels, method1, method2, 
                                      args.output_folder, use_absolute_numbers=False)
                plot_confusion_matrix(avg_cm, avg_cm_normalized, all_labels, method1, method2, 
                                      args.output_folder, use_absolute_numbers=True)
            elif args.output_type == 'absolute':
                plot_confusion_matrix(avg_cm, avg_cm_normalized, all_labels, method1, method2, 
                                      args.output_folder, use_absolute_numbers=True)
            else:  # 'relative'
                plot_confusion_matrix(avg_cm, avg_cm_normalized, all_labels, method1, method2, 
                                      args.output_folder, use_absolute_numbers=False)
                 
    print(f"Confusion matrices have been saved to '{args.output_folder}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Confusion Matrix Generator")
    parser.add_argument("--d_human_annotations", required=True, help="Path to D human annotations")
    parser.add_argument("--t_human_annotations", required=True, help="Path to T human annotations")
    parser.add_argument("--d_model", default="ArjanvD95/by_the_horns_D42G", help="Path to D model")
    parser.add_argument("--t_model", default="ArjanvD95/by_the_horns_T42G", help="Path to T model")
    parser.add_argument("--direct_demo1", required=True, help="Path to folder containing direct_demo1 predictions")
    parser.add_argument("--direct_demo2", required=True, help="Path to folder containing direct_demo2 predictions")
    parser.add_argument("--output_folder", default="confusion_matrices", help="Path to output folder for confusion matrices")
    parser.add_argument("--output_type", choices=['relative', 'absolute', 'both'], default='both', 
                        help="Type of confusion matrix to output: 'relative', 'absolute', or 'both' (default)")
    args = parser.parse_args()
    main(args)