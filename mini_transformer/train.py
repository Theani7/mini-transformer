import torch
import torch.nn.functional as F

from .model import MiniTransformer
from .vocab import encode, text, vocab_size

# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

if __name__ == "__main__":

    torch.manual_seed(42)

    data = torch.tensor(
        encode(text),
        dtype=torch.long
    )


    # Input and target

    x = data[:-1]
    y = data[1:]


    # Add batch dimension

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)


    # --------------------------------------------------
    # 2. Model
    # --------------------------------------------------

    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_len=64
    )


    # --------------------------------------------------
    # 3. Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )


    # --------------------------------------------------
    # 4. Training
    # --------------------------------------------------

    for step in range(1000):

        # Forward pass

        logits = model(x)

        # Calculate loss

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        # Clear old gradients

        optimizer.zero_grad()

        # Backpropagation

        loss.backward()

        # Update parameters

        optimizer.step()

        if step % 100 == 0:

            print(
                f"step {step:4d} | "
                f"loss {loss.item():.4f}"
            )

    torch.save(
        model.state_dict(),
        "model.pt"
    )

    print("\nsaved model.pt")
