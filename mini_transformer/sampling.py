import torch
import torch.nn.functional as F


def sample_next(
    logits,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    repetition_penalty=1.0,
    context=None,
):
    """Pick the next token id for each row of `logits`.

    `multinomial` returns a position in the *sorted* probability array, not a
    vocabulary id, so the sampled index must be mapped back through the sort
    permutation. Omitting that remap yields plausible garbage while greedy
    decoding from the same model is perfect.
    """
    if repetition_penalty != 1.0 and context is not None:
        logits = logits.clone()
        for b in range(logits.shape[0]):
            unique_tokens = context[b].unique()
            score = logits[b, unique_tokens]
            logits[b, unique_tokens] = torch.where(
                score > 0, score / repetition_penalty, score * repetition_penalty
            )

    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = F.softmax(logits / temperature, dim=-1)

    sorted_probs, perm = torch.sort(probs, dim=-1, descending=True)

    if top_k > 0:
        sorted_probs[:, top_k:] = 0.0

    # drop tokens whose *preceding* cumulative probability already passed top_p
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs = sorted_probs.masked_fill(cumulative - sorted_probs > top_p, 0.0)

    index = torch.multinomial(
        sorted_probs / sorted_probs.sum(dim=-1, keepdim=True), num_samples=1
    )

    return perm.gather(-1, index)
