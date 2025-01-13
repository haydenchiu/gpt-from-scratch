import torch
from transformers import AutoTokenizer
from src.model.gpt import GPT, generate
# from src.data.load_data import get_char_mappings
from src.config.config import TRAINING_CONFIG, MODEL_CONFIG


if __name__ == "__main__":
    # Define model
    device  = TRAINING_CONFIG["device"]
    vocab_size = MODEL_CONFIG["vocab_size"]
    context_length = MODEL_CONFIG["context_length"]
    model_dim = MODEL_CONFIG["model_dim"]
    num_blocks = MODEL_CONFIG["num_blocks"]
    num_heads = MODEL_CONFIG["num_heads"]
    learning_rate = TRAINING_CONFIG["learning_rate"]
    save_path = TRAINING_CONFIG["save_path"]

    # model_dim, num_heads, vocab_size, context_length, num_blocks
    model = GPT(model_dim, num_heads, vocab_size, context_length, num_blocks).to(device)

    # Load the best model
    model.load_state_dict(torch.load(save_path, weights_only=True))

    # Example starting context: beginning tokens for generation
    starting_text = "A quick brown fox jumps over"
    new_chars = 500  # Number of characters to generate

    # Load your tokenizer and mappings
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["tokenizer_name"], clean_up_tokenization_spaces=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Use tokenizer.encode to handle encoding automatically
    context = torch.tensor([tokenizer.encode(starting_text)], dtype=torch.long).to(device)
    
    # Generate text
    generated_text = generate(model, new_chars, context, context_length, tokenizer)
    print("Generated text:\n", starting_text + generated_text)