# Locating factual associations in GPT using causal tracing.
import torch
from transformer_lens import HookedTransformer


# We have data samples of the form
# (s, r, o) where s is subject, r is relation, and o is object.

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

### clean run
with torch.no_grad():
    clean_logits, clean_run_cache = model.run_with_cache(tokens)
print(clean_logits.shape)

### corrupted run

noise_scale = 3 * get_embedding_variance(model)

s_token_len = len(tokenizer.encode(fact['s']))
noise = torch.randn((1, s_token_len, model.cfg.d_model)) # only noise the subject
noise = torch.cat([noise, torch.zeros((1, tokens.shape[-1] - s_token_len, model.cfg.d_model))], dim=1) # pad for relation
noise = noise * noise_scale

def add_noise(value, hook):
    return value + noise


noise_hooks = [(f'hook_embed', add_noise)]

with model.hooks(fwd_hooks=noise_hooks), torch.no_grad():
    corrupted_logits, corrupted_run_cache = model.run_with_cache(tokens)

print(corrupted_logits.shape)

# p(correct) for clean and corrupted runs
correct_token = tokenizer.encode(fact['o'])[0]


layer_to_patch = 0
position_to_patch = 0

def patch_h_hook(value, hook):
    # value shape is (batch_size, seq_len, hidden_size)
    print('in patch_h_hook')
    print(value.shape)
    return value

patch_hooks = [
    (f'hook_embed', add_noise),
    (f'blocks.{layer_to_patch}.hook_resid_post', patch_h_hook),
]

with model.hooks(fwd_hooks=patch_hooks), torch.no_grad():
    patched_logits, patched_run_cache = model.run_with_cache(tokens)


print(torch.softmax(clean_logits[0, -1], dim=-1)[correct_token])
print(torch.softmax(corrupted_logits[0, -1], dim=-1)[correct_token])
print(torch.softmax(patched_logits[0, -1], dim=-1)[correct_token])
