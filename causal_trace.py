# %%
# Locating factual associations in GPT using causal tracing.

from matplotlib import pyplot as plt
import numpy as np

import time
import torch
from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer
from utils import get_embedding_variance, analyze_patch
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


def corrupt_and_patch(tokens, corrupted_tokens=None, optimize_noise=False, noise_mult=3):
    # assert not (corrupted_tokens is not None and optimize_noise), 'Cannot optimize noise if corrupted_tokens is provided'

    noise_sd = noise_mult * torch.sqrt(get_embedding_variance(model))

    if optimize_noise:
        noise, noisy_tokens, noisy_emb = find_noise(model, tokens, list(range(s_token_len)), noise_sd)
        corrupted_tokens = torch.tensor(noisy_tokens).unsqueeze(0).to(device)
        ct(noisy_tokens)
    else:
        noise = torch.randn((1, s_token_len, model.cfg.d_model)) # only noise the subject
        noise = torch.cat([noise, torch.zeros((1, tokens.shape[-1] - s_token_len, model.cfg.d_model))], dim=1)
        noise = noise * noise_sd
        noise = noise.to(device)

    if corrupted_tokens is not None:
        tokens = corrupted_tokens

    def add_noise(value, hook):
        if corrupted_tokens is None:
            return value + noise
        else:  # do nothing if corrupted_tokens is provided
            return value

    noise_hooks = [(f'hook_embed', add_noise)]
    with model.hooks(fwd_hooks=noise_hooks), torch.no_grad():
        corrupted_logits = model(tokens, return_type='logits')

    n_layers = model.cfg.n_layers
    n_positions = tokens.shape[-1]

    results = []
    t0 = time.time()

    # Iterate through every layer and position
    for layer_to_patch in range(n_layers):
        for position_to_patch in range(n_positions):
            result = analyze_patch(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise, width=5)
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


def compute_and_plot(corrupted_tokens=None, optimize_noise=False, noise_mult=10, trials=10):

    # run n times and average
    h_diffs = []
    a_diffs = []
    m_diffs = []
    original_probs = []
    corrupted_probs = []
    for i in range(trials):
        original_prob, corrupted_prob, h_diff, a_diff, m_diff = corrupt_and_patch(
            tokens=tokens,
            corrupted_tokens=corrupted_tokens,
            optimize_noise=optimize_noise,
            noise_mult=noise_mult
        )
        h_diffs.append(h_diff)
        a_diffs.append(a_diff)
        m_diffs.append(m_diff)
        original_probs.append(original_prob)
        corrupted_probs.append(corrupted_prob)


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

ct(tokens)
print('tokens device:', tokens.device)
corrupted_tokens = tokenizer.encode("The Colosseum is located in", return_tensors='pt').to(device)
ct(corrupted_tokens)

compute_and_plot(corrupted_tokens=corrupted_tokens, optimize_noise=False, noise_mult=3, trials=1)
# %%
compute_and_plot(corrupted_tokens=None, optimize_noise=False, noise_mult=3, trials=5)
# %%
compute_and_plot(corrupted_tokens=None, optimize_noise=True, noise_mult=3, trials=5)



# %%
