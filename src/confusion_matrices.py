import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from itertools import product


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
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if key in data:
                    results.append(data[key])
                else:
                    print(f"Key '{key}' not found in line {i+1}: {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError in line {i+1}: {e.msg} (line: {line.strip()})")
            except Exception as e:
                print(f"Error processing line {i+1}: {e} (line: {line.strip()})")
    return results

def flatten_predictions(predictions):
    return [item for sublist in predictions for item in sublist]

def create_confusion_matrix(true_labels, pred_labels):
    all_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    
    # Avoid division by zero with a small epsilon
    epsilon = 1e-8
    sum_with_epsilon = cm.sum(axis=1)[:, np.newaxis] + epsilon
    
    cm_normalized = cm.astype('float') / sum_with_epsilon
    
    return cm, cm_normalized, all_labels

def plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, output_folder, use_absolute_numbers=False):
    plt.figure(figsize=(12, 10))
    if use_absolute_numbers:
        cm_int = np.round(cm).astype(int)
        sns.heatmap(cm_normalized, annot=cm_int, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {method1} vs {method2}')
    plt.xlabel(f'Predicted ({method2})')
    plt.ylabel(f'True ({method1})')
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'confusion_matrix_{method1}_vs_{method2}.png')
    plt.savefig(output_file)
    plt.close()

def main(args):
    D_hum_predictions = load_jsonl(args.d_human_annotations, 'tags')
    T_hum_predictions = load_jsonl(args.t_human_annotations, 'tags')
    texts = load_jsonl(args.d_human_annotations, 'text')
    D_predictor = NERPredictor(args.d_model)
    T_predictor = NERPredictor(args.t_model)
    D_lllmaaa_predictions = [D_predictor.predict(text) for text in texts]
    T_lllmaaa_predictions = [T_predictor.predict(text) for text in texts]
    data = {
        "D_LLLMaAA": D_lllmaaa_predictions,
        "T_LLLMaAA": T_lllmaaa_predictions,
        "D_hum": D_hum_predictions,
        "T_hum": T_hum_predictions,
        "direct_demo1": load_jsonl(args.direct_demo1_file, 'tags'),
        "direct_demo2": load_jsonl(args.direct_demo2_file, 'tags')
    }
    os.makedirs(args.output_folder, exist_ok=True)
    for method1, method2 in product(data.keys(), repeat=2):
        if method1 != method2:
            print(f"Generating confusion matrix for {method1} vs {method2}")
            flat_preds1 = flatten_predictions(data[method1])
            flat_preds2 = flatten_predictions(data[method2])
            try:
                cm, cm_normalized, labels = create_confusion_matrix(flat_preds1, flat_preds2)
                # Determine how to plot based on output_type
                if args.output_type == 'both':
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=True)
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=False)
                elif args.output_type == 'absolute':
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=True)
                elif args.output_type == 'relative':
                    plot_confusion_matrix(cm, cm_normalized, labels, method1, method2, args.output_folder, use_absolute_numbers=False)
            except Exception as e:
                print(f"Error creating confusion matrix: {e}")
    print(f"Confusion matrices have been saved to '{args.output_folder}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Confusion Matrix Generator")
    parser.add_argument("--d_human_annotations", required=True, help="Path to D human annotations file")
    parser.add_argument("--t_human_annotations", required=True, help="Path to T human annotations file")
    parser.add_argument("--d_model", default="ArjanvD95/by_the_horns_D42G", help="Path to D model")
    parser.add_argument("--t_model", default="ArjanvD95/by_the_horns_T42G", help="Path to T model")
    parser.add_argument("--direct_demo1_file", required=True, help="Path to file containing direct_demo1 predictions")
    parser.add_argument("--direct_demo2_file", required=True, help="Path to file containing direct_demo2 predictions")
    parser.add_argument("--output_folder", default="confusion_matrices", help="Path to output folder for confusion matrices")
    parser.add_argument("--output_type", choices=['relative', 'absolute', 'both'], default='both', help="Type of confusion matrix to output")
    args = parser.parse_args()
    main(args)
