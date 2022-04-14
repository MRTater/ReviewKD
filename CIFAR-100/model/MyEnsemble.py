import torch
import torch.nn as nn


class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, modelC):
        super(MyEnsemble, self).__init__()

        with open(modelA, 'rb') as f:
            self.modelA.load_state_dict(torch.load(f))
            # modelA.eval()
        with open(modelB, 'rb') as f:
            self.modelB.load_state_dict(torch.load(f))
            # modelB.eval()
        with open(modelC, 'rb') as f:
            self.modelC.load_state_dict(torch.load(f))
            # modelC.eval()

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)
        out = out1 + out2 + out3
        return torch.argmax(out, dim=1)

def ensemble(**kwargs):
    """
    Constructs a ensemble.
    """
    model = MyEnsemble(**kwargs)
    return model
