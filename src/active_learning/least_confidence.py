import torch
from torch import nn
from .strategy import Strategy
from .utils import ner_predict, compute_uncertainty

class LeastConfidence(Strategy):
    def __init__(self, annotator_config_name, pool_size, setting: str = 'knn', engine: str='gpt-3.5-turbo-0125',
                 reduction: str='mean'):
        super().__init__(annotator_config_name, pool_size, setting, engine)
        assert reduction in ['mean', 'sum', 'min']
        self.reduction = reduction

    def query(self, args, k, dataset, model):
        pool_indices = self._get_pool_indices() #changed this
        pool_features = [dataset[int(i)] for i in pool_indices] #changed this
        pool_logits = ner_predict(args, model, pool_features)
        uncertainties = compute_uncertainty(pool_logits, reduction=self.reduction)
        topk_indices = torch.topk(uncertainties, k=k).indices
        return topk_indices.tolist() 