# %%
# Locating factual associations in GPT using causal tracing.

from matplotlib import pyplot as plt
import numpy as np

import time
import torch
from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer
from utils import get_embedding_variance


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


def corrupt_and_patch(tokens, corrupted_tokens=None):
    if corrupted_tokens is not None:
        tokens = corrupted_tokens

    noise_scale = 3 * torch.sqrt(get_embedding_variance(model))
    noise = torch.randn((1, s_token_len, model.cfg.d_model)) # only noise the subject
    noise = torch.cat([noise, torch.zeros((1, tokens.shape[-1] - s_token_len, model.cfg.d_model))], dim=1) # pad for relation
    noise = noise * noise_scale
    noise = noise.to(device)

    def add_noise(value, hook):
        if corrupted_tokens is None:
            return value + noise
        else:  # do nothing if corrupted_tokens is provided
            return value

    noise_hooks = [(f'hook_embed', add_noise)]
    with model.hooks(fwd_hooks=noise_hooks), torch.no_grad():
        corrupted_logits = model(tokens, return_type='logits')

    def analyze_patch(model, tokens, layer_to_patch, position_to_patch):
        h_patch_place = f'blocks.{layer_to_patch}.hook_resid_post'
        att_patch_place = f'blocks.{layer_to_patch}.hook_attn_out'
        mlp_patch_place = f'blocks.{layer_to_patch}.hook_mlp_out'

        def patch_h_hook(value, hook):
            value[:, position_to_patch] = clean_run_cache[h_patch_place][:, position_to_patch]
            return value
        
        def patch_att_hook(value, hook):
            value[:, position_to_patch] = clean_run_cache[att_patch_place][:, position_to_patch]
            return value
        
        def patch_mlp_hook(value, hook):
            value[:, position_to_patch] = clean_run_cache[mlp_patch_place][:, position_to_patch]
            return value
        
        patch_hooks = [(f'hook_embed', add_noise), (h_patch_place, patch_h_hook)]
        attn_patch_hooks = [(f'hook_embed', add_noise), (att_patch_place, patch_att_hook)]
        mlp_patch_hooks = [(f'hook_embed', add_noise), (mlp_patch_place, patch_mlp_hook)]

        with model.hooks(fwd_hooks=patch_hooks), torch.no_grad():
            h_patched_logits = model(tokens, return_type='logits')
        
        with model.hooks(fwd_hooks=attn_patch_hooks), torch.no_grad():
            att_patched_logits = model(tokens, return_type='logits')
        
        with model.hooks(fwd_hooks=mlp_patch_hooks), torch.no_grad():
            mlp_patched_logits = model(tokens, return_type='logits')
        
        h_patch_prob = torch.softmax(h_patched_logits[0, -1], dim=-1)[target_token]
        att_patch_prob = torch.softmax(att_patched_logits[0, -1], dim=-1)[target_token]
        mlp_patch_prob = torch.softmax(mlp_patched_logits[0, -1], dim=-1)[target_token]

        return h_patch_prob, att_patch_prob, mlp_patch_prob


    n_layers = model.cfg.n_layers
    n_positions = tokens.shape[-1]

    results = []
    t0 = time.time()

    # Iterate through every layer and position
    for layer_to_patch in range(n_layers):
        for position_to_patch in range(n_positions):
            result = analyze_patch(model, tokens, layer_to_patch, position_to_patch)
            result = tuple(r.to('cpu') for r in result) # ugly move to cpu.
            results.append((layer_to_patch, position_to_patch, result))

    print(f'Finished in {time.time() - t0:.2f} seconds')

    original_prob = torch.softmax(clean_logits[0, -1], dim=-1)[target_token].to('cpu')
    corrupted_prob = torch.softmax(corrupted_logits[0, -1], dim=-1)[target_token].to('cpu')

    h_diff_matrix = np.zeros((n_positions, n_layers))
    a_diff_matrix = np.zeros((n_positions, n_layers))
    m_diff_matrix = np.zeros((n_positions, n_layers))
    for res in results:
        layer_to_patch, position_to_patch, (h, a, m) = res
        h_diff = np.abs(original_prob - h)
        a_diff = np.abs(original_prob - a)
        m_diff = np.abs(original_prob - m)
        h_diff_matrix[position_to_patch, layer_to_patch] = h_diff
        a_diff_matrix[position_to_patch, layer_to_patch] = a_diff
        m_diff_matrix[position_to_patch, layer_to_patch] = m_diff
    
    return original_prob, corrupted_prob, h_diff_matrix.T, a_diff_matrix.T, m_diff_matrix.T

# %%

ct(tokens)
print('tokens device:', tokens.device)
corrupted_tokens = tokenizer.encode("The Colosseum is located in", return_tensors='pt').to(device)
ct(corrupted_tokens)


# %%


# run n times and average
h_diffs = []
a_diffs = []
m_diffs = []
original_probs = []
corrupted_probs = []
for i in range(1):
    original_prob, corrupted_prob, h_diff, a_diff, m_diff = corrupt_and_patch(
        tokens=tokens,
        corrupted_tokens=corrupted_tokens,
    )
    h_diffs.append(h_diff)
    a_diffs.append(a_diff)
    m_diffs.append(m_diff)
    original_probs.append(original_prob)
    corrupted_probs.append(corrupted_prob)

# %%

# Average the differences
h_diff_matrix = np.mean(h_diffs, axis=0)
a_diff_matrix = np.mean(a_diffs, axis=0)
m_diff_matrix = np.mean(m_diffs, axis=0)
original_prob = np.mean(original_probs)
corrupted_prob = np.mean(corrupted_probs)

# Common settings for x-labels and color scale
x_labels = [str(i) if i != s_token_len - 1 else str(i) + '*' for i in range(tokens.shape[1])]
vmin_vmax = {'vmin': 0, 'vmax': np.abs(original_prob - corrupted_prob)}

# Create a figure with side-by-side subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

# Plot for h_diff
ax1.set_xticks(range(tokens.shape[1]))
ax1.set_xticklabels(x_labels)
# im1 = ax1.imshow(h_diff_matrix, aspect='auto', cmap='viridis', **vmin_vmax)
im1 = ax1.imshow(h_diff_matrix, aspect='auto', cmap='viridis')
ax1.invert_yaxis()
fig.colorbar(im1, ax=ax1, label='Difference')
ax1.set_ylabel('Layer')
ax1.set_xlabel('Position')
ax1.set_title('Original - H_Patched')

# Plot for a_diff
ax2.set_xticks(range(tokens.shape[1]))
ax2.set_xticklabels(x_labels)
# im2 = ax2.imshow(a_diff_matrix, aspect='auto', cmap='viridis', **vmin_vmax)
im2 = ax2.imshow(a_diff_matrix, aspect='auto', cmap='viridis')
ax2.invert_yaxis()
fig.colorbar(im2, ax=ax2, label='Difference')
ax2.set_ylabel('Layer')
ax2.set_xlabel('Position')
ax2.set_title('Original - A_Patched')

# Plot for m_diff
ax3.set_xticks(range(tokens.shape[1]))
ax3.set_xticklabels(x_labels)
# im3 = ax3.imshow(m_diff_matrix, aspect='auto', cmap='viridis', **vmin_vmax)
im3 = ax3.imshow(m_diff_matrix, aspect='auto', cmap='viridis')
ax3.invert_yaxis()
fig.colorbar(im3, ax=ax3, label='Difference')
ax3.set_ylabel('Layer')
ax3.set_xlabel('Position')
ax3.set_title('Original - M_Patched')


# Save the figure
plt.tight_layout()
plt.savefig('difference_plot.png', dpi=300, bbox_inches='tight')


# %%
