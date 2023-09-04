# causal tracing for mlps.
# Idea is to
# 1. find the timestep which, when patched results in greatest improvement.
# 2. then prune to find most important neurons.


from matplotlib import pyplot as plt
import numpy as np

import time
import torch
from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer
from utils import get_embedding_variance
from find_noise import find_noise


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = HookedTransformer.from_pretrained('gpt2-small', device=device)
model = HookedTransformer.from_pretrained('gpt2-xl', device=device)
model.eval()

tokenizer = model.tokenizer
ct = ColoredTokenizer(tokenizer)

fact = {
    's': 'The Eiffel Tower',
    'r': ' is located in',
    'o': ' Paris'
}

prompt = fact['s'] + fact['r']
print(prompt)

tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
target_token = tokenizer.encode(fact['o'])[0]
s_token_len = len(tokenizer.encode(fact['s']))

with torch.no_grad():
    clean_logits, clean_run_cache = model.run_with_cache(tokens)


def patch_mlp_pos(model, tokens, position_to_patch, clean_run_cache, target_token, add_noise_fn):
    mlp_patch_places = [f'blocks.{layer}.hook_mlp_out' for layer in range(model.cfg.n_layers)]

    cached_mlp_outs = [clean_run_cache[place] for place in mlp_patch_places]
    cached_mlp_outs = torch.stack(cached_mlp_outs, dim=0)

    def patch_mlp_hook(value, hook):
        value[:, position_to_patch] = cached_mlp_outs[:, position_to_patch]
        return value
    
    mlp_patch_hooks = [(f'hook_embed', add_noise_fn), (mlp_patch_places, patch_mlp_hook)]
    with model.hooks(fwd_hooks=mlp_patch_hooks), torch.no_grad():
        mlp_patched_logits = model(tokens, return_type='logits')

    mlp_patch_prob = torch.softmax(mlp_patched_logits[0, -1], dim=-1)[target_token]
    return mlp_patch_prob



