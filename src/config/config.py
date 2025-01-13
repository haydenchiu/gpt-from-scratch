import torch

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MODEL_CONFIG = {
    "model_name": "gpt2",
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.85,
    "vocab_size": 50257,
    "context_length": 512,
    "model_dim": 768,
    "num_blocks": 12,
    "num_heads": 12,
    "tokenizer_name": 'gpt2'
}

TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 3e-5,
    "num_epochs": 1,
    "seed": 42,
    "device": device,
    "save_path": "gpt_model.pth",
    "valid_size": 0.1,
    "test_size": 0.1,
    "train_size": 0.001
}
