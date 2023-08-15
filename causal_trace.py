# Locating factual associations in GPT using causal tracing.
import torch
from transformer_lens import HookedTransformer


# We have data samples of the form
# (s, r, o) where s is subject, r is relation, and o is object.

model = HookedTransformer.from_pretrained('gpt2-small')
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
clean_logits, clean_run_cache = model.run_with_cache(tokens)
print(clean_logits.shape)

# corrupted run
# We add noise to the input embedding of the subject.

noise_scale = 3 * get_embedding_variance(model)

def add_noise(value, hook):
    noise = torch.randn_like(value)
    return value + noise * noise_scale

