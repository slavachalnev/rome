
def get_embedding_variance(model):
    """Compute unbiased variance of the embedding layer."""
    embedding = model.W_E.detach().clone().cpu()
    embedding_mean = embedding.mean(dim=0)
    embedding_variance = ((embedding - embedding_mean) ** 2).mean(dim=0)
    return embedding_variance

