# Locating factual associations in GPT using causal tracing.

from matplotlib import pyplot as plt
import numpy as np
import imageio

import time
import torch
from transformer_lens import HookedTransformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = HookedTransformer.from_pretrained('gpt2-small', device=device)
model = HookedTransformer.from_pretrained('gpt2-xl', device=device)
model.eval()

tokenizer = model.tokenizer

def get_embedding_variance(model):
    """Compute unbiased variance of the embedding layer."""
    embedding = model.W_E
    embedding_mean = embedding.mean(dim=0)
    embedding_variance = ((embedding - embedding_mean) ** 2).mean(dim=0)
    return embedding_variance


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


def corrupt_and_patch():
    noise_scale = 3 * torch.sqrt(get_embedding_variance(model))

    noise = torch.randn((1, s_token_len, model.cfg.d_model)) # only noise the subject
    noise = torch.cat([noise, torch.zeros((1, tokens.shape[-1] - s_token_len, model.cfg.d_model))], dim=1) # pad for relation
    noise = noise.to(device)
    noise = noise * noise_scale

    def add_noise(value, hook):
        return value + noise

    noise_hooks = [(f'hook_embed', add_noise)]
    with model.hooks(fwd_hooks=noise_hooks), torch.no_grad():
        # corrupted_logits, corrupted_run_cache = model.run_with_cache(tokens)
        corrupted_logits = model(tokens, return_type='logits')

    def analyze_patch(model, tokens, layer_to_patch, position_to_patch):
        patch_place = f'blocks.{layer_to_patch}.hook_resid_post'

        def patch_h_hook(value, hook):
            value[:, position_to_patch] = clean_run_cache[patch_place][:, position_to_patch]
            return value

        patch_hooks = [(f'hook_embed', add_noise), (patch_place, patch_h_hook)]

        with model.hooks(fwd_hooks=patch_hooks), torch.no_grad():
            # patched_logits, patched_run_cache = model.run_with_cache(tokens)
            patched_logits = model(tokens, return_type='logits')

        return torch.softmax(patched_logits[0, -1], dim=-1)[target_token]


    n_layers = model.cfg.n_layers
    n_positions = tokens.shape[-1]

    results = []
    t0 = time.time()

    # Iterate through every layer and position
    for layer_to_patch in range(n_layers):
        for position_to_patch in range(n_positions):
            result = analyze_patch(model, tokens, layer_to_patch, position_to_patch).to('cpu')
            results.append((layer_to_patch, position_to_patch, result))

    print(f'Finished in {time.time() - t0:.2f} seconds')

    original_prob = torch.softmax(clean_logits[0, -1], dim=-1)[target_token].to('cpu')
    corrupted_prob = torch.softmax(corrupted_logits[0, -1], dim=-1)[target_token].to('cpu')

    diff_matrix = np.zeros((n_positions, n_layers))
    for res in results:
        layer_to_patch, position_to_patch, patched_prob = res
        diff = np.abs(original_prob - patched_prob)
        diff_matrix[position_to_patch, layer_to_patch] = diff
    
    return diff_matrix.T, original_prob, corrupted_prob



"""
# # Initialize writer object
# writer = imageio.get_writer('difference_plot.gif', duration=500)

# for i in range(5):
#     diff_matrix, original_prob, corrupted_prob = corrupt_and_patch()
#     # Plotting the matrix
#     y_labels = [str(i) if i != s_token_len - 1 else str(i) + '*' for i in range(tokens.shape[1])]
#     plt.yticks(range(tokens.shape[1]), y_labels)
#     plt.imshow(diff_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=np.abs(original_prob - corrupted_prob))
#     plt.colorbar(label='Difference')
#     plt.xlabel('Layer')
#     plt.ylabel('Position')
#     plt.title('Original - Patched')

#     # Save plot to a temporary file and append to the GIF
#     plt.savefig('temp_plot.png', dpi=300, bbox_inches='tight')
#     writer.append_data(imageio.imread('temp_plot.png'))
#     plt.close()

# # Close the writer
# writer.close()
"""


# run 10 times and average
diff_matrices = []
original_probs = []
corrupted_probs = []
for i in range(10):
    diff_matrix, original_prob, corrupted_prob = corrupt_and_patch()
    diff_matrices.append(diff_matrix)
    original_probs.append(original_prob)
    corrupted_probs.append(corrupted_prob)

diff_matrix = np.mean(diff_matrices, axis=0)
original_prob = np.mean(original_probs)
corrupted_prob = np.mean(corrupted_probs)

x_labels = [str(i) if i != s_token_len - 1 else str(i) + '*' for i in range(tokens.shape[1])]
plt.xticks(range(tokens.shape[1]), x_labels)
plt.imshow(diff_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=np.abs(original_prob - corrupted_prob))
plt.gca().invert_yaxis()
plt.colorbar(label='Difference')
plt.ylabel('Layer')
plt.xlabel('Position')
plt.title('Original - Patched')
plt.savefig('difference_plot.png', dpi=300, bbox_inches='tight')

