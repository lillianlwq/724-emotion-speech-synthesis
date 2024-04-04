def get_VQVAE_Config():
    return {
        "batch_size": 256,
        "num_training_updates": 15000,
        "num_hiddens": 128,
        "num_residual_hiddens": 32,
        "num_residual_layers": 2,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "commitment_cost": 0.25,
        "decay": 0.99,
        "learning_rate": 1e-3
    }

def get_Audio_Config():
    return {
        "audio_path": "../ESD/English",
        "segments": 5,
        "sample_rate": 22050
    }