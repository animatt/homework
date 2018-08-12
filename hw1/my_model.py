import torch


class MyModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super().__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        out_relu = self.linear1(x).clamp(min=0)
        out = self.linear2(out_relu)

        return out
