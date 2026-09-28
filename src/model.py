import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

CATEGORIES = ['Political', 'Religious', 'Gender', 'Personal Offense',
              'Abusive/Violence', 'Origin', 'Body Shaming', 'Misc']


class DualHeadClassifier(nn.Module):
    """Transformer encoder with two outputs on the first token:
    one hate/not-hate logit and one logit per hate category."""

    def __init__(self, model_name, num_categories=len(CATEGORIES), dropout=0.1, pretrained=True):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            # used when loading our own fine-tuned weights afterwards
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden, 1)
        self.category_head = nn.Linear(hidden, num_categories)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.binary_head(cls).squeeze(-1), self.category_head(cls)


def load_trained(model_name, weights_path, device='cpu'):
    """Rebuild the model and load our fine-tuned weights (used by the explainability and demo notebooks)."""
    model = DualHeadClassifier(model_name, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()
