import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional
from functools import partial


class Interaction(nn.Module):

    def __init__(self,
                 in_features=2,
                 out_features=1,
                 kernel_size=(3, 3),
                 padding=(1, 1, 2, 0)):
        super(Interaction, self).__init__()

        self.conv = nn.Conv2d(in_features, out_features, kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_features)
        self.pad = partial(functional.pad, pad=padding)

    def forward(self, features_1, features_2):
        """
        Args:
            features_1: [B, T, F]
            features_2: [B, T, F]
        Return:
            [B, T, F]
        """

        old_features, features_to_add = features_1.unsqueeze(1), features_2.unsqueeze(1)
        net_input = torch.cat((old_features, features_to_add), dim=1)
        net_input = self.pad(net_input)
        mask = torch.sigmoid(self.batch_norm(self.conv(net_input))).squeeze()
        new_features_1 = features_1 + mask*features_2
        return new_features_1


if __name__ == "__main__":
    input_1, input_2 = torch.ones((2, 3, 2, 6), device='cuda:0')
    interacton_model = Interaction()
    interacton_model.to('cuda:0')
    output = interacton_model.interaction(input_1, input_2)
    print(1)