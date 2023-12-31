from typing import List

import torch
import torch.nn.functional as F
from torch.optim import Adam

from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer

from utils import get_embedding_variance

def find_noise(
        model: HookedTransformer,
        tokens: torch.Tensor,
        noise_idxs: List,
        noise_sd: float,
        steps: int = 100,
    ):
    """Find noise which minimizes loss of tokens[:max(noise_idxs) + 1]
    while keeping the standard deviation of the noise equal to noise_sd.
    """
    ct = ColoredTokenizer(model.tokenizer)

    # we only add noise to noise_idxs
    noise = torch.randn((len(noise_idxs), model.cfg.d_model))
    noise = noise * noise_sd
    original_norm = noise.norm(dim=-1)
    original_norm = original_norm.to(model.cfg.device)
    remaining_zeros = torch.zeros((tokens.shape[-1] - len(noise_idxs), model.cfg.d_model))
    noise = torch.cat([noise, remaining_zeros], dim=0)

    mask = torch.zeros_like(noise)
    mask[:len(noise_idxs), :] = 1
    mask = mask.to(model.cfg.device)

    noise_sd = noise_sd.to(model.cfg.device)
    noise = noise.to(model.cfg.device)
    noise.requires_grad = True

    optimizer = Adam([noise], lr=0.1)

    def add_noise(value, hook):
        return noise
    noise_hooks = [(f'hook_embed', add_noise)]

    for step in range(steps):
        optimizer.zero_grad()
        with model.hooks(fwd_hooks=noise_hooks):
            logits = model(tokens, return_type='logits')
        
        # loss averaged over all tokens after noise_idxs
        targets = tokens[0, max(noise_idxs) + 1:].to(model.cfg.device)
        loss = F.cross_entropy(logits[0, max(noise_idxs):-1], target=targets)

        noise_norm_term = ((noise[:len(noise_idxs)].norm(dim=-1) - original_norm) ** 2).mean()

        noisy_tokens, _ = similar_tokens(model, tokens, max(noise_idxs), noise)
        with model.hooks(fwd_hooks=noise_hooks):
            implausibility = model(torch.tensor(noisy_tokens).unsqueeze(0), return_type='loss')

        # if step % 10 == 0:
        #     gen = model.generate(torch.tensor(noisy_tokens).unsqueeze(0))
        #     ct(gen)
        #     print(f'step: {step}, loss: {loss.item():.2f}, noise_norm_term: {noise_norm_term.item():.2f}')
        #     print(f'implausibility: {implausibility.item():.2f}')

        loss = loss + noise_norm_term + 0.1*implausibility

        loss.backward()
        noise.grad *= mask
        optimizer.step()
    
    noise = noise.detach()

    # find the tokens which are most similar to noisy

    noisy_tokens, noisy_embeddings = similar_tokens(model, tokens, max(noise_idxs), noise)
    
    return noise, noisy_tokens, noisy_embeddings

def similar_tokens(model, tokens, last_noise_idx, noise):
    """Find the tokens which are most similar to noisy"""
    mask = torch.ones_like(noise)
    mask[:last_noise_idx + 1, :] = 0
    mask = mask.to(model.cfg.device)

    noisy_embeddings = model.embed(tokens)*mask + noise

    noisy_tokens = []
    for v in noisy_embeddings[0]:
        similarities = F.cosine_similarity(v.unsqueeze(0), model.W_E, dim=-1)
        best = similarities.argmax()
        noisy_tokens.append(best.item())
    
    return noisy_tokens, noisy_embeddings



if __name__ == '__main__':
    # model = HookedTransformer.from_pretrained('gpt2-small')
    model = HookedTransformer.from_pretrained('gpt2-xl')
    ct = ColoredTokenizer(model.tokenizer)

    sent = "The Eiffel Tower is located in"
    tokens = model.tokenizer.encode(sent, return_tensors='pt')
    ct(tokens)

    noise_idxs = [0, 1, 2, 3, 4]
    noise_mult = 1
    noise_sd = noise_mult * torch.sqrt(get_embedding_variance(model))

    noise, noisy_tokens, noisy_embeddings = find_noise(model, tokens, noise_idxs, noise_sd)

    # ct(noisy_tokens)
    pred = model.generate(torch.tensor(noisy_tokens).unsqueeze(0))
    ct(pred)
