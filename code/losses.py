import torch.nn as nn


class KDLoss(nn.Module):
    def __init__(self, temperature):
        super(KDLoss, self).__init__()
        self.temp = temperature
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, targets):
        return (
            -self.temp**2
            * (self.softmax(targets / self.temp) * self.log_softmax(preds / self.temp))
            .sum(dim=-1)
            .mean()
        )


class KDLossProb(nn.Module):
    def __init__(self, temperature):
        super(KDLossProb, self).__init__()
        self.temp = temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, target_probs):
        return (
            -self.temp**2
            * (target_probs * self.log_softmax(preds / self.temp)).sum(dim=-1).mean()
        )
