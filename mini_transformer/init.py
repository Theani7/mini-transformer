from torch import nn

STD = 0.02


def init_weights(model, n_layer):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=STD)

    residual_std = STD / (2 * n_layer) ** 0.5

    for block in model.blocks:
        nn.init.normal_(block.attention.proj.weight, mean=0.0, std=residual_std)
        nn.init.normal_(block.feed_forward.down.weight, mean=0.0, std=residual_std)

    return model
