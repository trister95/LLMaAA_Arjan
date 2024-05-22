# This script implements the loop of active learning + query from gpt.
# Each time a model is trained based on current annotated data.
import argparse
import os
import copy
import ujson as json
import numpy as np
import torch
import warnings
from environs import Env

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

from .data.processor import Processor
from .active_learning import RandomSampling, EntropySampling, LeastConfidence, KMeansSampling
from .train_ner import train_ner
from .utils import ugly_log, set_seed


def get_opt():
    parser = argparse.ArgumentParser()
    # data related 
    parser.add_argument('--dataset', default='en_conll03', type=str)
    # file related
    base_dir = os.path.dirname(os.path.realpath(__file__))
    default_save_path = os.path.join(base_dir, '..', 'models')
    print(default_save_path,"=default_save_path")
    
    parser.add_argument('--save_path', default=default_save_path, type=str, help="Path to save the model.")
    parser.add_argument('--load_path', default='', type=str)    # haven't implemented yet!
    # model related
    parser.add_argument('--model_name_or_path', default='bert-base-cased', type=str)
    # optimization related
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='The initial learning rate for bert layer.')
    #parser.add_argument('--adam_epsilon', default=1e-6, type=float,
                        #help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm.')
    parser.add_argument('--warmup_ratio', default=0.06, type=float,
                        help='Warm up ratio for Adam.')
    # training steps; use min between the two
    parser.add_argument('--num_train_epochs', default=10, type=int) #40
    parser.add_argument('--max_train_steps', default=10000, type=int)
    parser.add_argument('--early_stopping_patience', default=5, type=int)
    # active learning related
    parser.add_argument('--quadratic_selection', action='store_true', default=False,
                        help='Whether to use quadratic selection strategy.')
    parser.add_argument('--budget', default=500, type=int)          # for quadratic selection, fix a total budget
    parser.add_argument('--init_samples', default=50, type=int)
    parser.add_argument('--acquisition_samples', default=50, type=int)
    parser.add_argument('--acquisition_time', default=9, type=int)
    parser.add_argument('--strategy', default='confidence', type=str)
    # annotator related
    parser.add_argument('--engine', default='gpt-3.5-turbo', type=str)
    parser.add_argument('--annotator_config_name', default='en_conll03_base', type=str)
    parser.add_argument('--annotator_setting', default='knn', type=str,
                        help='The setting to retrieve demo.')
    # automatic reweighting strategy
    parser.add_argument('--reweight', action='store_true', default=True)
    # misc
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--notes', default='', type=str)
    # debug
    parser.add_argument('--store_track', action='store_true', default=False)
    #push_to_hub
    parser.add_argument('--push_to_hub', action='store_true', default=True,
                        help='Whether to push the model to Hugging Face Hub.')
    parser.add_argument('--model_name_on_hub', default='ArjanvD95/a_model_i_should_have_given_a_proper_name', type=str,
                        help='Model name on Hugging Face Hub.')
    return parser.parse_args()

def active_learning_loop(args):
    # get log
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.dirname(path)
    if not os.path.exists(os.path.join(path, f'logs/{args.dataset}')):
        os.mkdir(os.path.join(path, f'logs/{args.dataset}'))
    path = os.path.join(path, f'logs/{args.dataset}/{args.strategy}-{args.notes}.log')
    args.log_file = path
    # get device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    # get task type
    task_type = 'ner'
    # get config, tokenizer & data processor
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    cache_name = f'cache_{args.annotator_setting}_{args.engine}'
    data_processor = Processor(dataset=args.dataset, tokenizer=tokenizer, cache_name=cache_name)
    if task_type == 'ner':
        config.id2label = data_processor.get_id2tag()
        config.label2id = data_processor.get_tag2id()
        config.num_labels = len(config.id2label) 

    # add config from args 
    config.model_name_or_path = args.model_name_or_path
    # get data
    pool_features = data_processor.get_features(split='train')
    #pool_features = pool_features.shuffle(seed=42).select(range(int(0.1 * len(pool_features)))) #edit this out after testing
    print(len(pool_features),"=len(pool_features)")
    dev_features = data_processor.get_features(split='demo')
    test_features = data_processor.get_features(split='test')
    assert args.strategy in ['random', 'entropy', 'confidence', 'kmeans', 'hybrid']
    reduction = 'sum' if args.dataset == 'en_conll03' else 'mean'
    # reduction = 'sum'
    if args.strategy == 'random':
        strategy = RandomSampling(args.annotator_config_name, len(pool_features),
                                  args.annotator_setting, args.engine)
    elif args.strategy == 'entropy':
        strategy = EntropySampling(args.annotator_config_name, len(pool_features),
                                   args.annotator_setting, args.engine,
                                   reduction)  # sum for en_conll03
    elif args.strategy == 'confidence':    
        strategy = LeastConfidence(args.annotator_config_name, len(pool_features),
                                   args.annotator_setting, args.engine,
                                   reduction)  # sum for en_conll03
    elif args.strategy == 'kmeans':
        strategy = KMeansSampling(args.annotator_config_name, len(pool_features),
                                  args.annotator_setting, args.engine)
    else:
        raise ValueError('Unknown method.')
    # compute num of init samples
    if args.quadratic_selection:
        factor = int(args.budget / ((args.acquisition_time + 1) ** 2))
        n_init_samples = factor
    else:
        n_init_samples = args.init_samples
    indices = strategy.init_labeled_data(n_sample=n_init_samples)
    print(indices, "<--indices")
    records = strategy.update(indices, pool_features)
    print(records, "<--records")

    if len(records) > 0:
        print("more than zero records")
        data_processor.update_cache(records)
        data_processor.reload()
        pool_features = data_processor.get_features(split='train')
    active_learning_iterator = range(args.acquisition_time + 1)
    # begin active learning loop

    # get model
    if task_type == 'ner':
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=config.num_labels, id2label=config.id2label, label2id=config.label2id)
        model.to(device)
        train = train_ner
    else:
        raise ValueError('Unsupported task type.')

    for i in active_learning_iterator:
        print('========== begin active learning loop {} =========='.format(i))
        ugly_log(args.log_file, '========== begin active learning loop {} =========='.format(i))
        # get features
        train_features = strategy.get_labeled_data(pool_features)
        
        print(f'# of training data: {len(train_features)}')
        # debug
        if args.store_track:
            if i == 0:
                batch_indices = copy.deepcopy(strategy.lab_data_mask)
            else:
                batch_indices = np.logical_xor(strategy.lab_data_mask, batch_indices)
            batch_features = [pool_features[i] for i in np.where(batch_indices)[0] if pool_features[i]['labels'] is not None]
            batch_indices = copy.deepcopy(strategy.lab_data_mask)

            path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.dirname(path)
            debug_file = os.path.join(path, f'logs/debug_{args.strategy}.jsonl')
            ugly_log(debug_file, f'========== data in loop {i} ==========')
            ugly_log(debug_file, str(len(train_features)))
            for f in batch_features:
                ugly_log(debug_file, json.dumps(f, ensure_ascii=False))

        
        model = train(args, train_features, dev_features, model, config.id2label, tokenizer)
        # acquire new data
        if i == args.acquisition_time:
            continue
        print('========== acquiring new data ==========')
        # compute num of init samples
        if args.quadratic_selection:
            factor = int(args.budget / ((args.acquisition_time + 1) ** 2))
            k = factor * (2 * i + 3)
            if i == args.acquisition_time - 1:
                k = args.budget - factor * ((args.acquisition_time) ** 2)
        else:
            k = args.acquisition_samples
        indices = strategy.query(args, k, pool_features, model)
        print("indices acquired", indices)
        records = strategy.update(indices, pool_features)
        print("records acquired", records)
        if len(records) > 0:
            data_processor.update_cache(records)
            data_processor.reload()
            pool_features = data_processor.get_features(split='train')



if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="seems not to NE-tag")
    env = Env()
    env.read_env(".env")
    OPENAI_API_KEY = env.str("OPENAI_API_KEY")

    args = get_opt()
    print("Arguments parsed successfully.")

    set_seed(args.seed)
    active_learning_loop(args)