def KV_cache_calculation(B, S, n_layers, n_heads, d_head):
    # B = batch size
    # S = sequence length
    # n_layers = no of layers
    # n_heads = no of heads in each layer
    # d_head = dimension of each head = d_model / n_heads

    return 2 * B * S * n_layers * n_heads * d_head * 2 # 2 bytes per element

Total_cache_memory = KV_cache_calculation(1, 2048, 16, 8, 64)
Total_cache_memory = Total_cache_memory / 2**20
print(f"Total cache memory required: {Total_cache_memory} Megabytes")