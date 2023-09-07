import torch


def get_embedding_variance(model):
    """Compute unbiased variance of the embedding layer."""
    embedding = model.W_E.detach().clone().cpu()
    embedding_mean = embedding.mean(dim=0)
    embedding_variance = ((embedding - embedding_mean) ** 2).mean(dim=0)
    return embedding_variance


def analyze_patch_h(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise_fn):
    h_patch_place = f'blocks.{layer_to_patch}.hook_resid_post'

    def patch_h_hook(value, hook):
        value[:, position_to_patch] = clean_run_cache[h_patch_place][:, position_to_patch]
        return value
    
    patch_hooks = [(f'hook_embed', add_noise_fn), (h_patch_place, patch_h_hook)]

    with model.hooks(fwd_hooks=patch_hooks), torch.no_grad():
        h_patched_logits = model(tokens, return_type='logits')
    
    h_patch_prob = torch.softmax(h_patched_logits[0, -1], dim=-1)[target_token]

    return h_patch_prob


def analyze_patch_att(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise_fn, width=0):
    def get_hook(layer):
        att_patch_place = f'blocks.{layer}.hook_attn_out'
        def patch_att_hook(value, hook):
            value[:, position_to_patch] = clean_run_cache[att_patch_place][:, position_to_patch]
            return value
        return att_patch_place, patch_att_hook
    
    attn_patch_hooks = [(f'hook_embed', add_noise_fn)]
    min_layer = max(0, layer_to_patch - width)
    max_layer = min(model.cfg.n_layers, layer_to_patch + width + 1)
    for layer in range(min_layer, max_layer):
        attn_patch_hooks.append(get_hook(layer))

    with model.hooks(fwd_hooks=attn_patch_hooks), torch.no_grad():
        att_patched_logits = model(tokens, return_type='logits')
    
    att_patch_prob = torch.softmax(att_patched_logits[0, -1], dim=-1)[target_token]

    return att_patch_prob


def analyze_patch_mlp(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise_fn, width=0):

    def get_hook(layer):
        mlp_patch_place = f'blocks.{layer}.hook_mlp_out'
        def patch_mlp_hook(value, hook):
            value[:, position_to_patch] = clean_run_cache[mlp_patch_place][:, position_to_patch]
            return value
        return mlp_patch_place, patch_mlp_hook
    
    # mlp_patch_hooks = [(f'hook_embed', add_noise_fn), (mlp_patch_place, patch_mlp_hook)]
    mlp_patch_hooks = [(f'hook_embed', add_noise_fn)]
    min_layer = max(0, layer_to_patch - width)
    max_layer = min(model.cfg.n_layers, layer_to_patch + width + 1)
    for layer in range(min_layer, max_layer):
        mlp_patch_hooks.append(get_hook(layer))
    
    with model.hooks(fwd_hooks=mlp_patch_hooks), torch.no_grad():
        mlp_patched_logits = model(tokens, return_type='logits')
    
    mlp_patch_prob = torch.softmax(mlp_patched_logits[0, -1], dim=-1)[target_token]

    return mlp_patch_prob


def analyze_patch(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise_fn, width=0):
    h_patch_prob = analyze_patch_h(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise_fn)
    att_patch_prob = analyze_patch_att(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise_fn, width=width)
    mlp_patch_prob = analyze_patch_mlp(model, tokens, layer_to_patch, position_to_patch, clean_run_cache, target_token, add_noise_fn, width=width)
    return h_patch_prob, att_patch_prob, mlp_patch_prob
