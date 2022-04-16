import torch
import torch.nn as nn

from .resnet_cifar import build_resnet_backbone
from .reviewkd import build_review_kd


class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, modelC, KD_ensemble):
        super(MyEnsemble, self).__init__()
        self.KD_ensemble = KD_ensemble
        if KD_ensemble is None:
            with open(modelA, 'rb') as f:
                self.modelA = build_resnet_backbone(depth=int(32), num_classes=100)
                self.modelA.load_state_dict(torch.load(f))
                self.modelA.eval()
            with open(modelB, 'rb') as f:
                self.modelB = build_resnet_backbone(depth=int(32), num_classes=100)
                self.modelB.load_state_dict(torch.load(f))
                self.modelB.eval()
            with open(modelC, 'rb') as f:
                self.modelC = build_resnet_backbone(depth=int(32), num_classes=100)
                self.modelC.load_state_dict(torch.load(f))
                self.modelC.eval()
        else:
            with open(modelA, 'rb') as f:
                self.modelA = build_review_kd('resnet32', num_classes=100, teacher='resnet32')
                self.modelA.load_state_dict(torch.load(f))
                self.modelA.eval()
            with open(modelB, 'rb') as f:
                self.modelB = build_review_kd('resnet32', num_classes=100, teacher='resnet32')
                self.modelB.load_state_dict(torch.load(f))
                self.modelB.eval()
            with open(modelC, 'rb') as f:
                self.modelC = build_review_kd('resnet32', num_classes=100, teacher='resnet32')
                self.modelC.load_state_dict(torch.load(f))
                self.modelC.eval()

    def forward(self, x):
        if self.KD_ensemble is None:
            out1 = self.modelA(x)
            out2 = self.modelB(x)
            out3 = self.modelC(x)
            out = out1 + out2 + out3
            return torch.argmax(out, dim=1)
        else:
            results1, out1 = self.modelA(x)
            results2, out2 = self.modelB(x)
            results3, out3 = self.modelC(x)
            out = out1 + out2 + out3
            return torch.argmax(out, dim=1)

def ensemble(**kwargs):
    """
    Constructs a ensemble.
    """
    model = MyEnsemble(**kwargs)
    return model
