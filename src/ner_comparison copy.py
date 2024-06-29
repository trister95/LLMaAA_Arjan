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
    y_true_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    
    labels = sorted(set(y_true_flat + y_pred_flat))
    
    return f1_score(y_true_flat, y_pred_flat, labels=labels, average='weighted')

def pairwise_ner_comparison(pred_labels_list, method_names, output_csv):
    n_methods = len(method_names)
    results_mean = np.zeros((n_methods, n_methods))
    results_std = np.zeros((n_methods, n_methods))

    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i != j:
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
    D_predictor = NERPredictor(args.d_model)
    T_predictor = NERPredictor(args.t_model)

    # Load human annotations (which also serve as holdout data)
    D_hum_predictions = load_jsonl(args.human_annotations, 'tags')
    T_hum_predictions = load_jsonl(args.human_annotations, 'tags')
    texts = [json.loads(line)['text'] for line in open(args.human_annotations)]

    # Predict using LLLMaAA models (5 times each)
    D_lllmaaa_predictions = [D_predictor.predict(text) for _ in range(5) for text in texts]
    T_lllmaaa_predictions = [T_predictor.predict(text) for _ in range(5) for text in texts]

    # Load other predictions
    def load_multiple_files(folder_path):
        file_pattern = os.path.join(folder_path, '*.jsonl')
        return [load_jsonl(file) for file in glob(file_pattern)]

    D_llm_predictions = load_multiple_files(args.d_llm_predictions)
    T_llm_predictions = load_multiple_files(args.t_llm_predictions)
    direct_demo1_predictions = load_multiple_files(args.direct_demo1)
    direct_demo2_predictions = load_multiple_files(args.direct_demo2)

    pred_labels_list = {
        "D_LLLMaAA": D_lllmaaa_predictions,
        "T_LLLMaAA": T_lllmaaa_predictions,
        "D_llm": D_llm_predictions,
        "T_llm": T_llm_predictions,
        "D_hum": [D_hum_predictions],
        "T_hum": [T_hum_predictions],
        "direct_demo1": direct_demo1_predictions,
        "direct_demo2": direct_demo2_predictions
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
    parser.add_argument("--d_model", default="ArjanvD95/by_the_horns_D42G", help="Path to D model")
    parser.add_argument("--t_model", default="ArjanvD95/by_the_horns_T42G", help="Path to T model")
    parser.add_argument("--human_annotations", required=True, help="Path to human annotations (also serves as holdout data)")
    parser.add_argument("--d_llm_predictions", required=True, help="Path to folder containing D LLM predictions")
    parser.add_argument("--t_llm_predictions", required=True, help="Path to folder containing T LLM predictions")
    parser.add_argument("--direct_demo1", required=True, help="Path to folder containing direct_demo1 predictions")
    parser.add_argument("--direct_demo2", required=True, help="Path to folder containing direct_demo2 predictions")
    parser.add_argument("--output_csv", default="comparison_results.csv", help="Path to output CSV file")
    parser.add_argument("--output_mean", default="results_mean", help="Path to output mean numpy array (without .npy extension)")
    parser.add_argument("--output_std", default="results_std", help="Path to output std numpy array (without .npy extension)")
    
    args = parser.parse_args()
    main(args)