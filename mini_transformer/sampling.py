def sample_next(logits, temperature=1.0, top_p=1.0):
    return logits.argmax(dim=-1, keepdim=True)
