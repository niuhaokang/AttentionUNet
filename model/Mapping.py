import torch.nn as nn
import torch

class Mapping(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, feat):
        B, _, _, _ = feat.size()

if __name__ == '__main__':
    m = nn.AdaptiveAvgPool2d(1)
    input = torch.rand((4, 1024, 16, 16))
    out = m(input)
    out = out.flatten(2, 3).permute(0, 2, 1)
    print(out.size())
