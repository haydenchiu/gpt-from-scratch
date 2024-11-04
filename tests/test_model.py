import torch
from src.model.gpt import GPT, generate

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda device found.")
else:
    device = torch.device("cpu")
    print("cpu device found.")

# hyperparameters for testing
vocab_size = 104
context_length = 128
model_dim = 252
num_blocks = 6
num_heads = 6

model = GPT(
    model_dim=model_dim, 
    num_heads=num_heads, 
    vocab_size=vocab_size, 
    context_length=context_length, 
    num_blocks=num_blocks
).to(device)

new_chars = 500
context = torch.zeros(1, 3, dtype=torch.long, device=device)

# int_to_char = {
#     0: '\n', 1: ' ', 2: '!', 3: '"', 4: '$', 5: '%', 6: '&', 7: "'", 
#     8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 
#     15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 
#     22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 
#     29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 
#     36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 
#     43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 
#     50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 
#     57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 
#     64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 
#     71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 
#     78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 
#     85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë', 
#     92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 
#     98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'
#     }
int_to_char = {i: chr(65+i) for i in range(vocab_size)}

print(generate(model, new_chars, context, context_length, int_to_char))