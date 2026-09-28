import re

import numpy as np
import torch

# A "word" is a whitespace-separated token of clean_text. LIME, SHAP and the
# human rationale sheet all use this same definition, so their scores can be
# compared position by position.


def make_predict_fn(model, tokenizer, max_length, device, batch_size=64):
    """Return a function: list of texts -> probability of hate (binary head)."""
    @torch.no_grad()
    def predict_hate(texts):
        texts = [str(t) for t in texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                            padding=True, return_tensors='pt').to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                b, _ = model(enc['input_ids'], enc['attention_mask'])
            out.append(torch.sigmoid(b.float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    return predict_hate


def lime_word_weights(text, predict_hate, num_samples=1000, seed=42):
    """LIME weight for every word of text (positive = pushes towards hate)."""
    from lime.lime_text import LimeTextExplainer

    def proba(texts):
        p = predict_hate(texts)
        return np.column_stack([1 - p, p])

    explainer = LimeTextExplainer(class_names=['Non-hate', 'Hate'], split_expression=r'\s+',
                                  bow=False, random_state=seed)
    words = text.split()
    exp = explainer.explain_instance(text, proba, num_features=len(words),
                                     num_samples=num_samples, labels=(1,))
    lime_words = exp.domain_mapper.indexed_string.inverse_vocab
    assert list(lime_words) == words, 'LIME split the text differently from text.split()'
    weights = np.zeros(len(words))
    for pos, w in exp.as_map()[1]:
        weights[pos] = w
    return words, weights


def shap_word_weights(text, predict_hate, max_evals=500, seed=42):
    """SHAP value for every word of text (positive = pushes towards hate).
    Masked words are removed, the same way LIME removes them."""
    import shap

    # SHAP needs a non-empty mask token, so it writes a placeholder word that we
    # delete again before the model sees the text.
    placeholder = 'xxmaskxx'

    def predict_without_masked(texts):
        cleaned = [' '.join(re.sub(placeholder, ' ', str(t)).split()) for t in texts]
        return predict_hate(cleaned)

    masker = shap.maskers.Text(r'\s+', mask_token=placeholder, collapse_mask_token=False)
    explainer = shap.Explainer(predict_without_masked, masker, seed=seed)
    sv = explainer([text], max_evals=max_evals, silent=True)
    tokens = [str(t).strip() for t in sv.data[0]]
    values = np.asarray(sv.values[0], dtype=float)
    keep = [i for i, t in enumerate(tokens) if t]
    words = [tokens[i] for i in keep]
    assert words == text.split(), 'SHAP split the text differently from text.split()'
    return words, values[keep], float(np.asarray(sv.base_values).ravel()[0])


def top_words(words, weights, k=3):
    """Positions of the k words with the largest positive weight (towards hate)."""
    order = [i for i in np.argsort(-np.asarray(weights)) if weights[i] > 0]
    return order[:k]
