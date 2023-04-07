from collections import OrderedDict

import torch.nn as nn
import torch


class VisionTransformerFinetune(nn.Module):
    def __init__(self, vision_transformer, num_classes):
        super().__init__()
        self.hidden_dim = vision_transformer.hidden_dim
        self.representation_size = vision_transformer.representation_size

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.num_classes = num_classes

        self.encoder = vision_transformer.encoder

        self._process_input = vision_transformer._process_input

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if self.representation_size is None:
            heads_layers["head"] = nn.Linear(self.hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.hidden_dim, self.representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(self.representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
