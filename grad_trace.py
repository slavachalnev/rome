from matplotlib import pyplot as plt
import numpy as np

import time
import torch
from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer
from utils import get_embedding_variance
from find_noise import find_noise

# we are given:
# 1. position to patch
# 2. clean run cache (mlp post activations)
# 3. noise hooks
# 4. target token
# 5. model
# 6. tokens

# First patch clean mlp activations into the corrupted run.
# Then compute the gradient of the target token wrt the (patched) mlp activations

# compute importance of activations by calculating (clean - corrupted) * grad

# then prune the mlp activations that are not important for the target token
#   by setting them to the corrupted mlp activations.


pos_to_patch = 3


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
    mlp_patch_places = [f'blocks.{layer}.mlp.hook_post' for layer in range(model.cfg.n_layers)]

    cached_mlp_outs = [clean_run_cache[place] for place in mlp_patch_places]
    cached_mlp_outs = [cached[:, position_to_patch] for cached in cached_mlp_outs]
    cached_mlp_outs = [cached.detach().requires_grad_(True) for cached in cached_mlp_outs]

    def patch_mlp_hook(value, hook):
        layer = int(hook.name.split('.')[1])
        value[:, position_to_patch] = cached_mlp_outs[layer]
        return value
    
    mlp_patch_hooks = [(f'hook_embed', add_noise_fn)]
    for layer, place in enumerate(mlp_patch_places):
        mlp_patch_hooks.append((place, patch_mlp_hook))

    with model.hooks(fwd_hooks=mlp_patch_hooks):
        mlp_patched_logits = model(tokens, return_type='logits')

    mlp_patch_prob = torch.softmax(mlp_patched_logits[0, -1], dim=-1)[target_token]
    mlp_patch_prob.backward()

    grads = [cached.grad for cached in cached_mlp_outs]

    return mlp_patch_prob, grads


noise_sd = 3 * torch.sqrt(get_embedding_variance(model))
noise = torch.randn((1, s_token_len, model.cfg.d_model)) # only noise the subject
noise = torch.cat([noise, torch.zeros((1, tokens.shape[-1] - s_token_len, model.cfg.d_model))], dim=1)
noise = noise * noise_sd
noise = noise.to(device)

def add_noise(value, hook):
    return value + noise

# corrupted run
noise_hooks = [(f'hook_embed', add_noise)]
with model.hooks(fwd_hooks=noise_hooks), torch.no_grad():
    corrupted_logits, corrupted_run_cache = model.run_with_cache(tokens)

clean_prob = torch.softmax(clean_logits[0, -1], dim=-1)[target_token]
corrupted_prob = torch.softmax(corrupted_logits[0, -1], dim=-1)[target_token]

mlp_patch_prob, grads = patch_mlp_pos(model, tokens, pos_to_patch, clean_run_cache, target_token, add_noise)

# clean activations
clean_mlp_outs = [clean_run_cache[f'blocks.{layer}.mlp.hook_post'] for layer in range(model.cfg.n_layers)]
clean_mlp_outs = [cached[:, pos_to_patch] for cached in clean_mlp_outs]

# corrupted activations
corrupted_mlp_outs = [corrupted_run_cache[f'blocks.{layer}.mlp.hook_post'] for layer in range(model.cfg.n_layers)]
corrupted_mlp_outs = [cached[:, pos_to_patch] for cached in corrupted_mlp_outs]

# compute importance of activations by calculating (clean - corrupted) * grad

importance = []
for clean, corrupted, grad in zip(clean_mlp_outs, corrupted_mlp_outs, grads):
    importance.append((clean - corrupted) * grad)


# find top k activations with highest importance
k = 1000

# Flatten and concatenate all importance tensors
flat_importance = torch.cat([imp.flatten() for imp in importance])

# Sort the flattened tensor in descending order and pick top k indices
top_k_indices = torch.topk(flat_importance, k).indices

# To find out which layer and position each index corresponds to, you can use divmod.
layer_positions = [(i // model.cfg.d_mlp, i % model.cfg.d_mlp) for i in top_k_indices]

# print(f'Top {k} most important activations:')
# for layer, position in layer_positions:
#     print(f'Layer {layer} position {position}, importance {importance[layer][0][position]}')

# patch only the important activations
def patch_important(model, tokens, position_to_patch, important, clean_run_cache, target_token, add_noise_fn):
    mlp_patch_places = [f'blocks.{layer}.mlp.hook_post' for layer in range(model.cfg.n_layers)]

    cached_mlp_outs = [clean_run_cache[place] for place in mlp_patch_places]

    def patch_mlp_hook(value, hook):
        layer = int(hook.name.split('.')[1])
        if important[layer].numel() == 0:
            return value
        new_value = value.clone()
        new_value[:, position_to_patch, important[layer]] = cached_mlp_outs[layer][:, position_to_patch, important[layer]]
        return new_value
    
    mlp_patch_hooks = [(f'hook_embed', add_noise_fn)]
    for layer, place in enumerate(mlp_patch_places):
        mlp_patch_hooks.append((place, patch_mlp_hook))
    
    with model.hooks(fwd_hooks=mlp_patch_hooks), torch.no_grad():
        mlp_patched_logits = model(tokens, return_type='logits')
    
    mlp_patch_prob = torch.softmax(mlp_patched_logits[0, -1], dim=-1)[target_token]
    return mlp_patch_prob


important = [[] for _ in range(model.cfg.n_layers)]
for layer, position in layer_positions:
    important[layer].append(position)

important_tensors = []
for layer in range(model.cfg.n_layers):
    important_tensors.append(torch.tensor(important[layer]))

precision_patched_prob = patch_important(model, tokens, pos_to_patch, important_tensors, clean_run_cache, target_token, add_noise)
print(f'Clean prob: {clean_prob}')
print(f'Patched prob: {mlp_patch_prob}')
print(f'Precision patched prob: {precision_patched_prob}')
print(f'Corrupted prob: {corrupted_prob}')


