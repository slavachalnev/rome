from typing import List

import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer

from utils import get_embedding_variance

def find_noise(
        model: HookedTransformer,
        tokens: torch.Tensor,
        noise_idxs: List,
        noise_sd: float,
    ):
    """Find noise which minimizes loss of tokens[:max(noise_idxs) + 1]
    while keeping the standard deviation of the noise equal to noise_sd.
    """

    # we only add noise to noise_idxs
    noise = torch.randn((len(noise_idxs), model.cfg.d_model))
    noise = noise * noise_sd
    remaining_zeros = torch.zeros((tokens.shape[-1] - len(noise_idxs), model.cfg.d_model))
    noise = torch.cat([noise, remaining_zeros], dim=0)

    noise = noise.to(model.cfg.device)
    noise.requires_grad = True

    def add_noise(value, hook):
        return value + noise

    noise_hooks = [(f'hook_embed', add_noise)]
    with model.hooks(fwd_hooks=noise_hooks):
        logits = model(tokens, return_type='logits')
    
    # loss averaged over all tokens after noise_idxs
    targets = tokens[0, max(noise_idxs) + 1:].to(model.cfg.device)
    ct(targets)

    loss = F.cross_entropy(logits[0, max(noise_idxs):-1], target=targets)
    print(loss)
    


if __name__ == '__main__':
    model = HookedTransformer.from_pretrained('gpt2-small')
    ct = ColoredTokenizer(model.tokenizer)

    sent = "The Eiffel Tower is located in"
    tokens = model.tokenizer.encode(sent, return_tensors='pt')
    ct(tokens)

    noise_idxs = [0, 1, 2, 3, 4]
    noise_sd = 3 * torch.sqrt(get_embedding_variance(model))

    find_noise(model, tokens, noise_idxs, noise_sd)
    
