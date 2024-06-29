import argparse
import json
import csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
from glob import glob

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
    with open(file_path) as f:
        return [json.loads(line)[key] for line in f]

def calculate_weighted_f1(y_true, y_pred):
    # Debug prints
    print("y_true (first few):", y_true[:2])
    print("y_pred (first few):", y_pred[:2])

    # Ensure we're working with flat lists of strings
    y_true_flat = [str(item) for sublist in y_true for item in sublist]
    y_pred_flat = [str(item) for sublist in y_pred for item in sublist]
    
    print("y_true_flat (first few):", y_true_flat[:5])
    print("y_pred_flat (first few):", y_pred_flat[:5])

    # Get unique labels
    labels = sorted(set(y_true_flat) | set(y_pred_flat))
    
    print("Unique labels:", labels)

    return f1_score(y_true_flat, y_pred_flat, labels=labels, average='weighted')

def pairwise_ner_comparison(pred_labels_list, method_names, output_csv):
    n_methods = len(method_names)
    results_mean = np.zeros((n_methods, n_methods))
    results_std = np.zeros((n_methods, n_methods))

    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            print(f"we are now comparing {method1} and {method2}")
            scores = []
            for pred1_list in pred_labels_list[method1]:
                for pred2_list in pred_labels_list[method2]:
                    scores.append(calculate_weighted_f1(pred1_list, pred2_list))
            results_mean[i, j] = np.mean(scores)
            results_std[i, j] = np.std(scores)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method'] + method_names)

        for i, method in enumerate(method_names):
            row_mean = [method] + [f"{score:.4f}" for score in results_mean[i]]
            row_std = ['±'] + [f"{score:.4f}" for score in results_std[i]]
            writer.writerow(row_mean)
            writer.writerow(row_std)

    return results_mean, results_std

def main(args):
    # Load data
    predictor1 = NERPredictor(args.model1)
    predictor2 = NERPredictor(args.model2)

    # Load human annotations (which also serve as holdout data)
    hum1_predictions = load_jsonl(args.human_annotations1, 'tags')
    hum2_predictions = load_jsonl(args.human_annotations2, 'tags')
    texts = [json.loads(line)['text'] for line in open(args.human_annotations1)]

    # Predict using LLLMaAA models (5 times each)
    lllmaaa1_predictions = [[predictor1.predict(text) for text in texts] for _ in range(5)]
    lllmaaa2_predictions = [[predictor2.predict(text) for text in texts] for _ in range(5)]

    #write the variable llmaaa1_predictions to txt file
    with open('llmaaa1_predictions.txt', 'w') as f:
        f.write(str(lllmaaa1_predictions))

    # Load other predictions
    def load_multiple_files(folder_path):
        file_pattern = os.path.join(folder_path, '*.jsonl')
        return [load_jsonl(file) for file in glob(file_pattern)]

    direct_no_demo_predictions = load_multiple_files(args.direct_no_demo)
    with open('direct_no_demo.txt', 'w') as f:
        f.write(str(direct_no_demo_predictions))

    direct_demo1_predictions = load_multiple_files(args.direct_demo1)
    direct_demo2_predictions = load_multiple_files(args.direct_demo2)

    with open('direct_with_demo.txt', 'w') as f:
        f.write(str(direct_demo1_predictions))

    pred_labels_list = {
        "hum1": [hum1_predictions],
        "hum2": [hum2_predictions],
        "direct_no_demo": direct_no_demo_predictions,
        "direct_demo1": direct_demo1_predictions,
        "direct_demo2": direct_demo2_predictions,
        "indirect1": lllmaaa1_predictions,
        "indirect2": lllmaaa2_predictions
        }

    method_names = list(pred_labels_list.keys())

    results_mean, results_std = pairwise_ner_comparison(pred_labels_list, method_names, args.output_csv)

    print("\nWeighted F1 Score Matrix (Mean):")
    print(np.array2string(results_mean, precision=4, suppress_small=True))
    print("\nWeighted F1 Score Matrix (Standard Deviation):")
    print(np.array2string(results_std, precision=4, suppress_small=True))
    print("\n")

    print(f"Results have been saved to '{args.output_csv}'")

    # Save numpy arrays
    np.save(args.output_mean, results_mean)
    np.save(args.output_std, results_std)
    print(f"Mean results saved to '{args.output_mean}.npy'")
    print(f"Standard deviation results saved to '{args.output_std}.npy'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Comparison Tool")
    parser.add_argument("--model1", default="ArjanvD95/by_the_horns_D42G", help="Path to model1")
    parser.add_argument("--model2", default="ArjanvD95/by_the_horns_T42G", help="Path to model2")
    parser.add_argument("--human_annotations1", required=True, help="Path to first set of human annotations")
    parser.add_argument("--human_annotations2", required=True, help="Path to second set of human annotations")
    parser.add_argument("--direct_no_demo", required=True, help="Path to folder containing direct_no_demo predictions")
    parser.add_argument("--direct_demo1", required=True, help="Path to folder containing direct_demo1 predictions")
    parser.add_argument("--direct_demo2", required=True, help="Path to folder containing direct_demo2 predictions")
    parser.add_argument("--output_csv", default="comparison_results.csv", help="Path to output CSV file")
    parser.add_argument("--output_mean", default="results_mean", help="Path to output mean numpy array (without .npy extension)")
    parser.add_argument("--output_std", default="results_std", help="Path to output std numpy array (without .npy extension)")
    
    args = parser.parse_args()
    main(args)