import torch
from torch import nn
from .strategy import Strategy
from .utils import ner_predict

class EntropySampling(Strategy):
    def __init__(self, annotator_config_name, pool_size, setting: str='knn', engine: str='gpt-3.5-turbo-0125', 

                 reduction: str='mean'):
        super().__init__(annotator_config_name, pool_size, setting, engine)
        assert reduction in ['mean', 'sum', 'max']
        self.reduction = reduction

    def query(self, args, k: int, model: nn.Module, features):
        pool_indices = self._get_pool_indices()
        pool_features = [features[i] for i in pool_indices]
        if self.task_type == 'ner':
            pred_logits = ner_predict(args, pool_features, model)
            uncertainties = []
            for logit in pred_logits:
                prob = torch.softmax(logit, dim=-1)
                entropy = torch.special.entr(prob).sum(dim=-1) # entropy over each token
                if self.reduction == 'mean':
                    uncertainties.append(entropy.mean())
                elif self.reduction == 'sum':
                    uncertainties.append(entropy.sum())
                elif self.reduction == 'max':
                    uncertainties.append(entropy.max())
            uncertainties = torch.stack(uncertainties)
        else:
            raise ValueError('tbd.')
        lab_indices = torch.topk(uncertainties, k=k)[1]
        lab_indices = [pool_indices[i] for i in lab_indices]
        return lab_indices