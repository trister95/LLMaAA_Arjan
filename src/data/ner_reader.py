import os
import ujson as json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

MAX_LEN = 512

def convert_span_labels_to_sequence_labels(tokens, span_label, language):
    """
    Convert span labels to IOBES sequence labels, which are compatible with common NER evaluators.
    Language: en/zh
    """
    span_label = sorted(span_label, key=lambda x: len(x['span']), reverse=True)
    span_to_type = {entity['span']: entity['type'] for entity in span_label}
    
    if language == 'zh':
        text = ''.join(tokens)
        for entity in span_label:
            span = entity['span']
            text = ('[sep]' + span + '[sep]').join(text.split(span))
        words = text.split('[sep]')
    else:
        dictionary = dict()
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = f'[{len(dictionary)}]'
        id_string = ' '.join([dictionary[token] for token in tokens])
        for entity in span_label:
            span_tokens = entity['span'].strip().split(' ')
            if not all(token in dictionary for token in span_tokens):
                continue
            id_substring = ' '.join([dictionary[token] for token in span_tokens])
            id_string = ('[sep]' + id_substring + '[sep]').join(id_string.split(id_substring))
        sent = id_string
        for token in dictionary:
            sent = sent.replace(dictionary[token], token)
        words = sent.split('[sep]')

    labels = []
    for word in words:
        word = word.strip()
        if len(word) == 0:
            continue
        entity_flag = (word in span_to_type)
        word_length = len(word.split(' ')) if language == 'en' else len(word)

        if entity_flag:
            entity_type = span_to_type[word]
            if word_length == 1:
                labels.append(f'S-{entity_type}')
            else:
                labels.extend([f'B-{entity_type}'] + [f'I-{entity_type}'] * (word_length - 2) + [f'E-{entity_type}'])
        else:
            labels.extend(['O'] * word_length)

    assert len(labels) == len(tokens)
    return labels

def ner_reader(tokenizer: PreTrainedTokenizer, dataset: str, cache_name: str='', use_cache: bool=True):
    """
    Read the dataset, tokenize the data, and align the labels accordingly,
    converting it to a format suitable for training in the Hugging Face ecosystem.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text.
        dataset (str): The name of the dataset.
        tag2id (dict): Mapping from tag names to their corresponding ids.

    Returns:
        dict: A dictionary containing tokenized and aligned datasets split by train, demo, and test.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(os.path.dirname(dir_path))
    dir_path = os.path.join(dir_path, f'data/{dataset}')
    files = {key: os.path.join(dir_path, f'{key}.jsonl') for key in ['train', 'demo', 'test']}

    if use_cache:
        cache_file = os.path.join(dir_path, f'{cache_name}.json')
        print("cache_file", cache_file)
        cache = json.load(open(cache_file, 'r', encoding='utf-8'))

    meta_path = os.path.join(dir_path, 'meta.json')
    tag2id = json.load(open(meta_path, 'r'))['tag2id']            
    processed_data = {}
    for split, file_path in files.items():
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line.strip()) for line in file]
            # Prepare structure for from_dict format
            processed_data[split] = {
                'input_ids': [],
                'attention_mask': [],
                'labels': [],
                'id': [],
                'text': [],
                'tokens':[]
            }

            for sample in tqdm(data, desc=f"Processing {split} data"):
                tokens = sample['tokens']
                tags = sample['tags']
                tokenized_inputs = tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length', max_length=512)
                
                labels = []
                word_ids = tokenized_inputs.word_ids()
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        labels.append(-100)
                    elif word_idx != previous_word_idx:
                        labels.append(tag2id[tags[word_idx]])
                    else:
                        labels.append(-100)
                    previous_word_idx = word_idx
                            
                if split == 'train' and use_cache and sample['id'] not in cache: 
                    labels = None
                # Append each field separately to the correct list

                processed_data[split]['input_ids'].append(tokenized_inputs['input_ids'])
                processed_data[split]['attention_mask'].append(tokenized_inputs['attention_mask'])
                processed_data[split]['labels'].append(labels)
                processed_data[split]['id'].append(sample['id'])
                processed_data[split]['text'].append(sample['text'])
                processed_data[split]['tokens'].append(sample['tokens'])
    return processed_data


if __name__ == '__main__':
    tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-cased')
    dataset = ner_reader(tokenizer, 'en_conll03', 'cache_en_conll03', True)
    print(dataset['train'][0])