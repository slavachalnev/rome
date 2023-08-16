# Locating factual associations in GPT using causal tracing.
import time
import torch
from transformer_lens import HookedTransformer

# We have data samples of the form
# (s, r, o) where s is subject, r is relation, and o is object.


def analyze_patch(model, tokens, fact, noise, layer_to_patch, position_to_patch):
    def add_noise(value, hook):
        return value + noise

    noise_hooks = [(f'hook_embed', add_noise)]

    with model.hooks(fwd_hooks=noise_hooks), torch.no_grad():
        corrupted_logits, corrupted_run_cache = model.run_with_cache(tokens)

    patch_place = f'blocks.{layer_to_patch}.hook_resid_post'

    def patch_h_hook(value, hook):
        value[:, position_to_patch] = clean_run_cache[patch_place][:, position_to_patch]
        return value

    patch_hooks = [
        (f'hook_embed', add_noise),
        (patch_place, patch_h_hook),
    ]

    with model.hooks(fwd_hooks=patch_hooks), torch.no_grad():
        patched_logits, patched_run_cache = model.run_with_cache(tokens)

    correct_token = tokenizer.encode(fact['o'])[0]

    return (torch.softmax(clean_logits[0, -1], dim=-1)[correct_token],
            torch.softmax(corrupted_logits[0, -1], dim=-1)[correct_token],
            torch.softmax(patched_logits[0, -1], dim=-1)[correct_token])


model = HookedTransformer.from_pretrained('gpt2-small', device='cpu')
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

tokens = tokenizer.encode(prompt, return_tensors='pt')

# clean run
with torch.no_grad():
    clean_logits, clean_run_cache = model.run_with_cache(tokens)

### corrupted run

noise_scale = 3 * get_embedding_variance(model)

s_token_len = len(tokenizer.encode(fact['s']))
noise = torch.randn((1, s_token_len, model.cfg.d_model)) # only noise the subject
noise = torch.cat([noise, torch.zeros((1, tokens.shape[-1] - s_token_len, model.cfg.d_model))], dim=1) # pad for relation
noise = noise * noise_scale




# Storing the results
results = []
t0 = time.time()

# Iterate through every layer and position
for layer_to_patch in range(model.cfg.n_layers):
    for position_to_patch in range(tokens.shape[-1]):
        result = analyze_patch(model, tokens, fact, noise, layer_to_patch, position_to_patch)
        results.append((layer_to_patch, position_to_patch, result))

print(f'Finished in {time.time() - t0:.2f} seconds')

# Print or further analyze the results
for res in results:
    print(res)
