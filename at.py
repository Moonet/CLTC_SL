import torch
import torch.nn as nn
import torch.nn.functional as F


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='bert.embeddings.word_embeddings.'):
        # emb_name should be consistent with name in actual model
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)


    def restore(self, emb_name='bert.embeddings.word_embeddings.'):
        # emb_name should be consistent with name in actual model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


