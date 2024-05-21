import os
import time
import ujson as json
from func_timeout.exceptions import FunctionTimedOut
from openai import RateLimitError
from abc import ABC, abstractmethod
import numpy as np
from torch import nn
from ..llm_annotator import Annotator

RETRY = 3

class Strategy(ABC):
    def __init__(self, annotator_config_name, pool_size, setting: str='knn', engine: str='gpt-3.5-turbo-0125'):
        """
        Base class for active learning strategy.
        Functionality:
            - Maintain the labeled & pool data set with mask.
            - Query strategies are implemented in subclass, which return indices.
            - Annotate all data in labeled set.
        Notice:
            - Reloading is implemented & handled by data.Processor.
            - Load demo file and demo index w.r.t. train data.
        """
        self.lab_data_mask = np.zeros(pool_size, dtype=bool)
        self.annotator = Annotator(engine, annotator_config_name)
        self.dataset = self.annotator.dataset
        if self.dataset in ['en_conll03', 'zh_msra', 'zh_onto4']:
            self.task_type = 'ner'
        elif self.dataset in ['en_semeval10', 'en_retacred']:
            self.task_type = 're'
        else:
            raise ValueError('Unknown dataset.')
        self.setting = setting
        # use knn demo
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(os.path.dirname(dir_path))
        demo_file_path = os.path.join(dir_path, f'data/{self.dataset}/demo.jsonl')
        self.demo_file = dict()
        with open(demo_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.demo_file[sample['id']] = sample
        if setting == 'random' or setting == 'knn':
            demo_index_path = os.path.join(dir_path, f'data/{self.dataset}/train-{setting}-demo.json')
            self.demo_index = json.load(open(demo_index_path, 'r', encoding='utf-8'))
        elif setting == 'zero':
            pass
        else:
            raise ValueError(f'Unknown setting {setting}.')

    def __len__(self):
        return len(self.lab_data_mask)

    def _get_labeled_indices(self):
        return np.where(self.lab_data_mask)[0]
    
    def _get_pool_indices(self):
        return np.where(~self.lab_data_mask)[0]
    
    def get_labeled_data(self, features):
        labeled_indices = self._get_labeled_indices()
        labeled_data = features.select(labeled_indices)
        return labeled_data
    
    @abstractmethod
    def query(self, args, k, model, features):
        # return k indices
        # assume k <= len(pool)
        return
    
    def init_labeled_data(self, n_sample: int=None):
        if n_sample is None:
            raise ValueError('Please specify initial sample ratio/size.')
        assert n_sample <= len(self)

        indices = np.arange(len(self))
        np.random.shuffle(indices)
        indices = indices[: n_sample]
        self.lab_data_mask[indices] = True
        return indices
    
    def update(self, indices, features):
        self.lab_data_mask[indices] = True
        return self.annotate(features)
    
    def annotate(self, features):
        results = {}
        labeled_indices = self._get_labeled_indices()
        for i in labeled_indices:
            feature = features[int(i)]
            label_key = 'labels' if self.task_type == 'ner' else 'label_id'


            if feature[label_key] is None:  # need to be annotated
                # get demo if not `zero`
                if self.setting == 'random' or self.setting == 'knn':
                    # pointer: {'id': id, ('score': score)}
                    demo = [self.demo_file[pointer['id']] 
                        for pointer in reversed(self.demo_index[feature['id']])]
                else:
                    demo = None

                result = None
                for j in range(RETRY):
                    try:
                        result = self.annotator.online_annotate(feature, demo)
                        break
                    except FunctionTimedOut:
                        print('Timeout. Retrying...')
                    except RateLimitError:
                        print('Rate limit. Sleep for 60 seconds...')
                        time.sleep(60)

                results[feature['id']] = result
    
        print('Annotate {} new records.'.format(len(results)))
        return results
