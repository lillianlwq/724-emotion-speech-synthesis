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


def get_Transformer_Config():
    return {
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.1,
        "sequence_length": 64,
        "input_dim": 26,
        "output_dim": 25,
        "d_model":  128,
        "dim_feedforward": 512,
        "feature_size": 32,
        "batch_size": 64,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "experiment_name": "runs/tmodel",
        "checkpoint_path": "../ESD"
    }


def get_TransDS_Config():
    return {
        "origin_path": "../ESD/English",
        "TTS_path": "../ESD/TTS",
        "label_path": "../ESD/TTS/labels.txt"
    }


def get_Emotion_Id(emotion):
    emotion_to_id = {
        'Happy': 0,
        'Sad': 1,
        'Angry': 2,
        'Neutral': 3,
        'Surprise': 4
    }
    return emotion_to_id[emotion]
