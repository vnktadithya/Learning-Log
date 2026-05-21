def KV_cache_calculation(B, S, n_layers, n_heads, d_head):
    # B = batch size
    # S = sequence length
    # n_layers = no of layers
    # n_heads = no of heads in each layer
    # d_head = dimension of each head = d_model / n_heads

    return 2 * B * S * n_layers * n_heads * d_head * 2 # 2 bytes per element

Total_cache_memory = KV_cache_calculation(16, 4096, 32, 32, 128)
Total_cache_memory = Total_cache_memory / 2**30
print(f"Total cache memory required: {Total_cache_memory} Gigabytes")
Single_element_memory = KV_cache_calculation(1, 1, 32, 32, 128)
Single_element_memory = Single_element_memory / 2**10
print(f'byte footprint of a single token: {Single_element_memory} Kilobytes')

