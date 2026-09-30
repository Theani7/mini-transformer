import torch
import torch.nn.functional as F


def sample_next(logits, temperature=1.0, top_p=1.0):
    """Pick the next token id for each row of `logits`.

    `multinomial` returns a position in the *sorted* probability array, not a
    vocabulary id, so the sampled index must be mapped back through the sort
    permutation. Omitting that remap yields plausible garbage while greedy
    decoding from the same model is perfect.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = F.softmax(logits / temperature, dim=-1)

    sorted_probs, perm = torch.sort(probs, dim=-1, descending=True)

    # drop tokens whose *preceding* cumulative probability already passed top_p
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs = sorted_probs.masked_fill(cumulative - sorted_probs > top_p, 0.0)

    index = torch.multinomial(
        sorted_probs / sorted_probs.sum(dim=-1, keepdim=True), num_samples=1
    )

    return perm.gather(-1, index)
